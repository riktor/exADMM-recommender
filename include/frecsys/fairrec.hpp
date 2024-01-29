// Copyright 2022 ************
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Branched from
// https://github.com/google-research/google-research/tree/master/ials with
// modification.
#pragma once

#include <algorithm>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "frecsys/recommender.hpp"

namespace frecsys {

class FairRecRecommender : public Recommender {
public:
  FairRecRecommender(const std::string& user_embeddings_path,
                     const std::string& item_embeddings_path,
                     float min_assignment_prop, int topk_fair) {
    user_embedding_ = LoadMatrix(user_embeddings_path);
    item_embedding_ = LoadMatrix(item_embeddings_path);

    // Initialize embedding matrices
    num_items_ = item_embedding_.rows();
    embedding_dim_ = user_embedding_.cols();

    min_assignment_prop_ = min_assignment_prop;
    topk_fair_ = topk_fair;
  }

  VectorXf Score(const int user_id, const SpVector& user_history) override {
    throw("Function 'Score' is not implemented");
  }

  // Custom implementation of EvaluateDataset that does the projection using the
  // iterative optimization algorithm.
  EvaluationResult EvaluateDataset(const VectorXi& k_list, const Dataset& data,
                                   const SpMatrix& eval_by_user) override {
    std::unordered_map<int, int> user_to_ind;

    const int num_items = item_embedding_.rows();
    const int num_users = user_embedding_.rows();
    MatrixXf user_scores = MatrixXf::Zero(num_users, num_items);

    int cnt = 0;
    for (const auto& user_and_history : data.by_user()) {
      user_to_ind[user_and_history.first] = cnt;
      cnt++;
    }

    ComputeScores(
        data.by_user(), item_embedding_,
        [&](const int user_id) -> MatrixXf::RowXpr {
          return user_embedding_.row(user_to_ind[user_id]);
        },
        [&](const int user_id) -> MatrixXf::RowXpr {
          return user_scores.row(user_to_ind[user_id]);
        });

    // Evaluate the dataset.
    std::unordered_map<int, VectorXf> fair_scores = CreateDeterministicRanking(
        num_users, num_items_, data,
        [&](const int user_id) -> MatrixXf::RowXpr {
          return user_scores.row(user_to_ind[user_id]);
        },
        min_assignment_prop_);
    return EvaluateDatasetInternal(
        num_items, k_list, data, eval_by_user,
        [&](const int user_id, const SpVector& history) -> VectorXf {
          return fair_scores[user_id];
        },
        false);
  }

  template <typename F, typename G>
  void ComputeScores(const SpMatrix& data_by_user,
                     const MatrixXf& item_embedding, F get_user_embedding_ref,
                     G get_user_scores_ref) {
    std::mutex m;
    auto data_by_user_iter = data_by_user.begin();  // protected by m
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back(std::thread([&] {
        while (true) {
          // Get a new user to work on.
          m.lock();
          if (data_by_user_iter == data_by_user.end()) {
            m.unlock();
            return;
          }
          int u = data_by_user_iter->first;
          SpVector train_history = data_by_user_iter->second;
          VectorXf user_emb = get_user_embedding_ref(u);
          ++data_by_user_iter;
          m.unlock();

          VectorXf scores = item_embedding * user_emb;

          // set small scores for items to exclude
          for (const auto& item_and_rating_index : train_history) {
            scores(item_and_rating_index.first) =
                std::numeric_limits<float>::lowest();
          }

          m.lock();
          get_user_scores_ref(u) = scores;
          m.unlock();
        }
      }));
    }
    // Join all threads.
    for (auto& th : threads) {
      th.join();
    }
  }

  // Templated implementation for the greedy-round-robin algorithm.
  template <typename F>
  std::unordered_map<int, VectorXf> CreateDeterministicRanking(
      const int num_users, const int num_items, const Dataset& data,
      F get_user_scores_ref, const float alpha) {
    std::unordered_map<int, VectorXi> topk_rankings;
    std::unordered_map<int, VectorXf> fair_scores;
    std::unordered_map<int, std::vector<size_t>> sorted_indices;

    std::vector<float> item_max_assign(num_items);
    fill(item_max_assign.begin(), item_max_assign.end(), 0);
    for (auto train_history : data.by_item()) {
      item_max_assign[train_history.first] =
          num_users - train_history.second.size();
    }

    auto data_by_user = data.by_user();
    std::vector<int> users;
    for (auto train_history : data_by_user) {
      int u = train_history.first;
      topk_rankings[u] = VectorXi::Zero(num_items);
      fair_scores[u] = VectorXf::Zero(num_items);
      users.push_back(u);

      VectorXf scores = get_user_scores_ref(u);
      std::vector<size_t> topk(num_items);
      std::iota(topk.begin(), topk.end(), 0);
      std::stable_sort(
          topk.begin(), topk.end(),
          [&scores](size_t i1, size_t i2) { return scores[i1] > scores[i2]; });
      sorted_indices[u] = topk;
    }
    int min_assigned = ceil((num_users * topk_fair_ / num_items) * alpha);
    VectorXf item_max_assign_vec =
        Eigen::Map<VectorXf>(item_max_assign.data(), num_items);
    VectorXf stock = (VectorXf::Ones(num_items) * min_assigned)
                         .array()
                         .min(item_max_assign_vec.array());

    // first phase (greedy-round-robin).
    std::random_shuffle(users.begin(), users.end());
    int cur = 0;
    while (1) {
      int num_stocks = stock.sum();
      if (num_stocks == 0) break;

      int u = users[cur];
      if (topk_rankings[u].sum() == topk_fair_) break;

      VectorXf scores = get_user_scores_ref(u);
      int nearest_item = -1;
      float rank = topk_rankings[u].sum() + 1;
      for (int i = 0; i < num_items; ++i) {
        nearest_item = sorted_indices[u][i];
        if (topk_rankings[u](nearest_item) > 0 || stock(nearest_item) == 0) {
          continue;
        }
        break;
      }
      topk_rankings[u](nearest_item) = 1;
      fair_scores[u](nearest_item) = 1.0 / rank;
      stock(nearest_item) -= 1;
      cur = (cur + 1) % num_users;
    }

    // second phase.
    for (int j = 0; j < num_users; ++j) {
      int u = users[j];
      while (1) {
        if (topk_rankings[u].sum() == topk_fair_) break;
        VectorXf scores = get_user_scores_ref(u);
        int nearest_item = -1;
        float rank = topk_rankings[u].sum() + 1;
        for (int i = 0; i < num_items; ++i) {
          nearest_item = sorted_indices[u][i];
          if (topk_rankings[u](nearest_item) > 0) {
            continue;
          }
          break;
        }
        topk_rankings[u](nearest_item) = 1;
        fair_scores[u](nearest_item) = 1.0 / rank;
      }
    }

    return fair_scores;
  }

private:
  int num_items_;
  MatrixXf user_embedding_;
  MatrixXf item_embedding_;

  int embedding_dim_;
  float min_assignment_prop_;
  int topk_fair_;
};
}  // namespace frecsys

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
#include <fstream>
#include <mutex>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

#include <glog/logging.h>
#include "Eigen/Core"
#include "Eigen/Dense"
#include "frecsys/dataset.hpp"
#include "frecsys/evaluation.hpp"
#include "frecsys/types.hpp"

namespace frecsys {

class Recommender {
public:
  virtual ~Recommender() {}

  virtual VectorXf Score(const int user_id, const SpVector& user_history) {
    return VectorXf::Zero(1);
  }

  // Common implementation for evaluating a dataset. It uses the scoring
  // function of the class.
  virtual EvaluationResult EvaluateDataset(const VectorXi& k_list,
                                           const Dataset& data,
                                           const SpMatrix& eval_by_user);

  virtual void Train(const Dataset& dataset) {}
  virtual void SetPrintTrainStats(const bool print_trainstats){};
  virtual void SetSaveExposureDist(const bool save_exposure_dist){};
  virtual void SetSaveEmbeddings(const bool save_embeddings){};
  virtual void SaveMatrix(const MatrixXf& matrix, const std::string& file_name);
  virtual MatrixXf LoadMatrix(const std::string& file_name);

  void init_matrix(MatrixXf* matrix, std::mt19937& gen,
                   const float adjusted_stdev) {
    std::normal_distribution<float> d(0, adjusted_stdev);
    for (int i = 0; i < matrix->size(); ++i) {
      *(matrix->data() + i) = d(gen);
    }
  };

  // Evaluate a single user.
  UserEvaluationResult EvaluateUser(const int num_items, const VectorXi& k_list,
                                    const VectorXf& all_scores,
                                    const SpVector& ground_truth,
                                    const SpVector& exclude);

  // Templated implementation for evaluating a dataset. Requires a function that
  // scores all items for a given user or history.
  template <typename F>
  EvaluationResult EvaluateDatasetInternal(const int num_items,
                                           const VectorXi& k_list,
                                           const Dataset& data,
                                           const SpMatrix& eval_by_user,
                                           F score_user_and_history,
                                           const bool save_exposure_dist) {
    std::mutex m;
    auto eval_by_user_iter = eval_by_user.begin();  // protected by m
    int num_ks = k_list.size();
    VectorXf recall = VectorXf::Zero(num_ks);
    VectorXf ndcg = VectorXf::Zero(num_ks);
    MatrixXf exposure_uni = MatrixXf::Zero(num_ks, num_items);
    MatrixXf exposure_logrank = MatrixXf::Zero(num_ks, num_items);

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back(std::thread([&] {
        while (true) {
          // Get a new user to work on.
          m.lock();
          if (eval_by_user_iter == eval_by_user.end()) {
            m.unlock();
            return;
          }
          int u = eval_by_user_iter->first;
          SpVector ground_truth = eval_by_user_iter->second;
          ++eval_by_user_iter;
          m.unlock();

          // Process the user.
          const SpVector& user_history = data.by_user().at(u);
          VectorXf scores = score_user_and_history(u, user_history);
          UserEvaluationResult this_metrics = this->EvaluateUser(
              num_items, k_list, scores, ground_truth, user_history);
          m.lock();
          // Update the metric.
          recall.noalias() += this_metrics.recall;
          ndcg.noalias() += this_metrics.ndcg;
          exposure_uni.noalias() += this_metrics.exposure_uni;
          exposure_logrank.noalias() += this_metrics.exposure_logrank;
          m.unlock();
        }
      }));
    }
    // Join all threads.
    for (auto& th : threads) {
      th.join();
    }

    recall /= eval_by_user.size();
    ndcg /= eval_by_user.size();
    MatrixXf ones = MatrixXf::Ones(exposure_uni.rows(), exposure_uni.cols());
    MatrixXf zeros = MatrixXf::Zero(exposure_uni.rows(), exposure_uni.cols());
    VectorXf coverage = exposure_uni.array()
                            .min(ones.array())
                            .max(zeros.array())
                            .rowwise()
                            .mean();

    if (save_exposure_dist) {
      SaveMatrix(exposure_uni, "exposure_uniform.log");
      SaveMatrix(exposure_logrank, "exposure_logrank.log");
    }

    // Compute gini coefficient.
    auto gini = [](VectorXf& exposure) -> float {
      int num_items = exposure.size();
      float diffsum = 0;
      for (int i = 0; i < num_items; ++i) {
        diffsum += (exposure.array() - exposure[i]).abs().sum();
      }
      float am = exposure.array().mean();
      float md = diffsum / pow(num_items, 2.0);
      return (md / am) / 2.0;
    };

    // Compute gini coefficient for the uniform exposure model.
    VectorXf gini_uni = VectorXf::Zero(num_ks);
    for (int i = 0; i < num_ks; i++) {
      VectorXf exp = exposure_uni.row(i);
      gini_uni[i] = gini(exp);
    }

    // Compute gini coefficient for the logrank exposure model.
    VectorXf gini_logrank = VectorXf::Zero(num_ks);
    for (int i = 0; i < num_ks; i++) {
      VectorXf exp = exposure_logrank.row(i);
      gini_logrank[i] = gini(exp);
    }

    EvaluationResult result = {k_list,   recall,   ndcg,
                               coverage, gini_uni, gini_logrank};
    return result;
  }
};

UserEvaluationResult Recommender::EvaluateUser(const int num_items,
                                               const VectorXi& k_list,
                                               const VectorXf& all_scores,
                                               const SpVector& ground_truth,
                                               const SpVector& exclude) {
  VectorXf scores = all_scores;
  for (uint64_t i = 0; i < exclude.size(); ++i) {
    assert(exclude[i].first < scores.size());
    scores[exclude[i].first] = std::numeric_limits<float>::lowest();
  }

  // Compute top-K ranking.
  int max_k = k_list.maxCoeff();
  std::vector<size_t> topk(scores.size());
  std::iota(topk.begin(), topk.end(), 0);
  std::nth_element(
      topk.begin(), topk.begin() + max_k, topk.end(),
      [&scores](size_t i1, size_t i2) { return scores[i1] > scores[i2]; });
  std::stable_sort(
      topk.begin(), topk.begin() + max_k,
      [&scores](size_t i1, size_t i2) { return scores[i1] > scores[i2]; });

  // Compute Recall@K.
  auto recall = [](int k, const std::set<int>& gt_set,
                   const std::vector<size_t>& topk) -> float {
    double result = 0.0;
    for (int i = 0; i < k; ++i) {
      if (gt_set.find(topk[i]) != gt_set.end()) {
        result += 1.0;
      }
    }
    return result / std::min<float>(k, gt_set.size());
  };

  // Compute nDCG@K.
  auto ndcg = [](int k, const std::set<int>& gt_set,
                 const std::vector<size_t>& topk) -> float {
    double result = 0.0;
    for (int i = 0; i < k; ++i) {
      if (gt_set.find(topk[i]) != gt_set.end()) {
        result += 1.0 / log2(i + 2.0);
      }
    }
    double norm = 0.0;
    for (int i = 0; i < std::min<int>(k, gt_set.size()); ++i) {
      norm += 1.0 / log2(i + 2.0);
    }
    return result / norm;
  };

  // Construct the set of positive items.
  std::set<int> gt_set;
  std::transform(ground_truth.begin(), ground_truth.end(),
                 std::inserter(gt_set, gt_set.begin()),
                 [](const std::pair<int, int>& p) { return p.first; });

  int num_ks = k_list.size();
  VectorXf recall_res(num_ks);
  VectorXf ndcg_res(num_ks);
  for (int i = 0; i < num_ks; ++i) {
    recall_res(i) = recall(k_list(i), gt_set, topk);
    ndcg_res(i) = ndcg(k_list(i), gt_set, topk);
  }

  // Accumulate item exposure for the uniform exposure model.
  MatrixXf exposure = MatrixXf::Zero(num_ks, num_items);
  for (int i = 0; i < num_ks; ++i) {
    for (int j = 0; j < k_list(i); ++j) {
      exposure(i, topk[j]) = 1;
    }
  }
  // Accumulate item exposure for the logrank exposure model.
  MatrixXf exposure_logrank = MatrixXf::Zero(num_ks, num_items);
  for (int i = 0; i < num_ks; ++i) {
    for (int j = 0; j < k_list(i); ++j) {
      exposure_logrank(i, topk[j]) = 1.0 / log2(i + 2.0);
    }
  }
  UserEvaluationResult result = {recall_res, ndcg_res, exposure,
                                 exposure_logrank};
  return result;
}

EvaluationResult Recommender::EvaluateDataset(const VectorXi& k_list,
                                              const Dataset& data,
                                              const SpMatrix& eval_by_user) {
  return EvaluateDatasetInternal(
      data.max_item() + 1, k_list, data, eval_by_user,
      [&](const int user_id, const SpVector& history) -> VectorXf {
        return Score(user_id, history);
      },
      false);
}

// Seriarise an Eigen matrix object to CSV.
void Recommender::SaveMatrix(const MatrixXf& matrix,
                             const std::string& file_name) {
  const int num_rows = matrix.rows();
  const int num_cols = matrix.cols();
  std::ofstream ofs(file_name);
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      ofs << matrix(i, j);
      if (j != num_cols - 1) {
        ofs << ",";
      }
    }
    if (i != num_rows - 1) {
      ofs << std::endl;
    }
  }
  ofs.close();
}

// Load CSV as an Eigen matrix object.
MatrixXf Recommender::LoadMatrix(const std::string& file_name) {
  std::ifstream infile(file_name);
  std::string line;
  std::string selem;
  size_t rows = 0;
  std::vector<float> vec;
  while (std::getline(infile, line)) {
    std::istringstream instr(line);
    while (std::getline(instr, selem, ',')) {
      float elem = std::atof(selem.c_str());
      vec.push_back(elem);
    }
    ++rows;
  }
  infile.close();
  size_t cols = vec.size() / rows;
  MatrixXf mat = Eigen::Map<MatrixXf>(vec.data(), rows, cols);

  return mat;
}
}  // namespace frecsys

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

#include <chrono>
#include <mutex>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

#include <glog/logging.h>
#include "Eigen/Core"
#include "Eigen/Dense"

#include "frecsys/recommender.hpp"

namespace frecsys {

class EXADMMRecommender : public Recommender {
public:
  EXADMMRecommender(int embedding_dim, int num_users, int num_items, float reg,
                    float reg_exp, float reg_expo, float learning_rate,
                    float admm_penalty, float unobserved_weight, float stdev,
                    int pred_iterations)
      : user_embedding_(num_users, embedding_dim),
        item_embedding_(num_items, embedding_dim),
        user_embedding_avg_(embedding_dim), dual_embedding_(embedding_dim) {
    // Initialise variables
    float adjusted_stdev = stdev / sqrt(embedding_dim);
    std::random_device rd{};
    std::mt19937 gen{rd()};
    init_matrix(&user_embedding_, gen, adjusted_stdev);
    init_matrix(&item_embedding_, gen, adjusted_stdev);
    // Initialise s to the true user average embedding.
    user_embedding_avg_ = user_embedding_.colwise().mean();
    dual_embedding_ = VectorXf::Zero(embedding_dim);

    regularization_ = reg;
    regularization_exp_ = reg_exp;
    regularization_expo_ = reg_expo;
    learning_rate_ = learning_rate;
    admm_penalty_ = admm_penalty;

    pred_iterations_ = pred_iterations;

    embedding_dim_ = embedding_dim;
    unobserved_weight_ = unobserved_weight;

    save_exposure_dist_ = false;
    save_embeddings_ = false;
  }

  VectorXf Score(const int user_id, const SpVector& user_history) override {
    throw("Function 'Score' is not implemented");
  }

  inline static const VectorXf ProjectU(
      const int num_users, const SpVector& user_history,
      const VectorXf& user_emb, const MatrixXf& item_embeddings,
      const MatrixXf& gramian, const VectorXf& user_embedding_avg,
      const VectorXf& dual_embedding, const float reg,
      const float unobserved_weight, const float learning_rate,
      const float admm_penalty) {
    // Parallel part of the U step.
    assert(user_history.size() > 0);

    int embedding_dim = item_embeddings.cols();
    assert(embedding_dim > 0);

    // Copy the previous estimate.
    VectorXf new_value = user_emb;

    // Hessian matrix.
    MatrixXf matrix = unobserved_weight * gramian;
    for (int i = 0; i < embedding_dim; ++i) {
      matrix(i, i) += reg;
    }

    const int kMaxBatchSize = 128;
    auto matrix_symm = matrix.selfadjointView<Eigen::Lower>();
    VectorXf rhs = VectorXf::Zero(embedding_dim);
    const int batch_size =
        std::min(static_cast<int>(user_history.size()), kMaxBatchSize);
    int num_batched = 0;
    MatrixXf factor_batch(embedding_dim, batch_size);
    for (const auto& item_and_rating_index : user_history) {
      const int cp = item_and_rating_index.first;
      assert(cp < item_embeddings.rows());
      const VectorXf& cp_v = item_embeddings.row(cp);

      factor_batch.col(num_batched).noalias() = cp_v;
      rhs.noalias() += cp_v;

      ++num_batched;
      if (num_batched == batch_size) {
        matrix_symm.rankUpdate(factor_batch);
        num_batched = 0;
      }
    }
    if (num_batched != 0) {
      const auto& factor_block =
          factor_batch.block(0, 0, embedding_dim, num_batched);
      matrix_symm.rankUpdate(factor_block);
    }

    // Gradient descent for the iALS loss
    new_value.noalias() -= learning_rate * (matrix * user_emb - rhs);
    // Parallel part of proximal mapping
    new_value.noalias() += ((learning_rate * admm_penalty) / num_users) *
                           (user_embedding_avg - dual_embedding);
    return new_value;
  }

  inline static const VectorXf ProjectV(
      const SpVector& item_history, const MatrixXf& item_embedding,
      const MatrixXf& user_embeddings, const VectorXf& user_embedding_avg,
      const MatrixXf& gramian, const MatrixXf& gramian_expo, const float reg,
      const float reg_expo, const float unobserved_weight) {
    // The V step.
    assert(item_history.size() > 0);

    int embedding_dim = user_embeddings.cols();
    assert(embedding_dim > 0);

    // This part is different from the V step of iALS.
    // Hessian includes the Gramian of s.
    MatrixXf matrix = unobserved_weight * gramian + reg_expo * gramian_expo;
    for (int i = 0; i < embedding_dim; ++i) {
      matrix(i, i) += reg;
    }

    const int kMaxBatchSize = 128;
    auto matrix_symm = matrix.selfadjointView<Eigen::Lower>();
    VectorXf rhs = VectorXf::Zero(embedding_dim);
    const int batch_size =
        std::min(static_cast<int>(item_history.size()), kMaxBatchSize);
    int num_batched = 0;
    MatrixXf factor_batch(embedding_dim, batch_size);
    for (const auto& user_and_rating_index : item_history) {
      const int cp = user_and_rating_index.first;
      assert(cp < user_embeddings.rows());
      const VectorXf& cp_v = user_embeddings.row(cp);

      factor_batch.col(num_batched).noalias() = cp_v;
      rhs.noalias() += cp_v;

      ++num_batched;
      if (num_batched == batch_size) {
        matrix_symm.rankUpdate(factor_batch);
        num_batched = 0;
      }
    }
    if (num_batched != 0) {
      const auto& factor_block =
          factor_batch.block(0, 0, embedding_dim, num_batched);
      matrix_symm.rankUpdate(factor_block);
    }

    Eigen::LLT<MatrixXf, Eigen::Lower> cholesky(matrix);
    assert(cholesky.info() == Eigen::Success);
    return cholesky.solve(rhs);
  }

  EvaluationResult EvaluateDataset(const VectorXi& k_list, const Dataset& data,
                                   const SpMatrix& eval_by_user) override {
    std::unordered_map<int, int> user_to_ind;
    VectorXf prediction(data.num_tuples());

    MatrixXf user_embedding =
        MatrixXf::Zero(data.by_user().size(), embedding_dim_);

    // Initialise the user and predictions to 0.0. (Note: this code needs to
    // change if the embeddings would have biases).
    int num_users = 0;
    for (const auto& user_and_history : data.by_user()) {
      user_to_ind[user_and_history.first] = num_users;
      for (const auto& item_and_rating_index : user_and_history.second) {
        prediction.coeffRef(item_and_rating_index.second) = 0.0;
      }
      num_users++;
    }

    // Fit the embeddings of new users.
    MatrixXf item_gramian = item_embedding_.transpose() * item_embedding_;
    VectorXf user_embedding_avg = user_embedding.colwise().mean();
    VectorXf dual_embedding = VectorXf::Zero(embedding_dim_);
    float reg_expo = ExpoRegularizationValue(num_users);
    float admm_penalty = PenaltyValue(num_users);
    for (int i = 0; i < pred_iterations_; i++) {
      StepU(
          data.by_user(),
          [&](const int user_id) -> MatrixXf::RowXpr {
            return user_embedding.row(user_to_ind[user_id]);
          },
          user_embedding, item_embedding_, user_embedding_avg, dual_embedding,
          item_gramian, learning_rate_, admm_penalty);

      const VectorXf& user_embedding_avg_true = user_embedding.colwise().mean();

      StepAvgU(user_embedding_avg, user_embedding, item_embedding_,
               dual_embedding, user_embedding_avg_true, item_gramian, reg_expo,
               admm_penalty);

      StepDualVar(dual_embedding, user_embedding, user_embedding_avg,
                  user_embedding_avg_true, admm_penalty);
    }

    if (save_embeddings_) {
      SaveMatrix(user_embedding, "user_embeddings.csv");
      SaveMatrix(item_embedding_, "item_embeddings.csv");
    }

    // Evaluate the dataset.
    uint64_t num_items = item_embedding_.rows();
    return EvaluateDatasetInternal(
        num_items, k_list, data, eval_by_user,
        [&](const int user_id, const SpVector& history) -> VectorXf {
          return item_embedding_ *
                 user_embedding.row(user_to_ind[user_id]).transpose();
        },
        save_exposure_dist_);
  }

  void Train(const Dataset& data) override {
    int num_users = user_embedding_.rows();
    float reg_expo = ExpoRegularizationValue(num_users);
    float admm_penalty = PenaltyValue(num_users);

    // Optimise the item embeddings V.
    float residual_V = StepV(
        data.by_item(),
        [&](const int index) -> MatrixXf::RowXpr {
          return item_embedding_.row(index);
        },
        user_embedding_, item_embedding_, user_embedding_avg_, reg_expo,
        print_residualstats_);

    // Compute the item Gramian to reuse the following steps.
    MatrixXf item_gramian = item_embedding_.transpose() * item_embedding_;

    // Optimise the user embeddings U.
    float residual_U = StepU(
        data.by_user(),
        [&](const int index) -> MatrixXf::RowXpr {
          return user_embedding_.row(index);
        },
        user_embedding_, item_embedding_, user_embedding_avg_, dual_embedding_,
        item_gramian, learning_rate_, admm_penalty, print_residualstats_);

    // Compute the true average user embedding to reuse the following steps.
    VectorXf user_embedding_avg_true = user_embedding_.colwise().mean();

    // Optimise the average user embedding s.
    float residual_s =
        StepAvgU(user_embedding_avg_, user_embedding_, item_embedding_,
                 dual_embedding_, user_embedding_avg_true, item_gramian,
                 reg_expo, admm_penalty, print_residualstats_);

    // Optimise the dual variables w.
    float residual_w = StepDualVar(dual_embedding_, user_embedding_,
                                   user_embedding_avg_, user_embedding_avg_true,
                                   admm_penalty, print_residualstats_);
    ComputeLosses(data);

    if (print_residualstats_) {
      LOG(INFO) << fmt::format(
          "V residual: {0}, U residual: {1}, s residual: {2}, w residual: {3}",
          residual_V, residual_U, residual_s, residual_w);
    }
  }

  void ComputeLosses(const Dataset& data) {
    if (!print_trainstats_) {
      return;
    }

    auto time_start = std::chrono::steady_clock::now();
    // Predict the dataset.
    VectorXf prediction(data.num_tuples());
    for (const auto& user_and_history : data.by_user()) {
      VectorXf user_emb = user_embedding_.row(user_and_history.first);
      for (const auto& item_and_rating_index : user_and_history.second) {
        prediction.coeffRef(item_and_rating_index.second) =
            item_embedding_.row(item_and_rating_index.first).dot(user_emb);
      }
    }
    int num_items = item_embedding_.rows();
    int num_users = user_embedding_.rows();

    // Compute observed loss.
    float loss_observed = (prediction.array() - 1.0).matrix().squaredNorm();

    // Compute L2 regulariser.
    double loss_reg = 0.0;
    for (const auto& user_and_history : data.by_user()) {
      loss_reg +=
          user_embedding_.row(user_and_history.first).squaredNorm() *
          RegularizationValue(user_and_history.second.size(), num_items);
    }
    for (const auto& item_and_history : data.by_item()) {
      loss_reg +=
          item_embedding_.row(item_and_history.first).squaredNorm() *
          RegularizationValue(item_and_history.second.size(), num_users);
    }

    // Unobserved loss.
    MatrixXf user_gramian = user_embedding_.transpose() * user_embedding_;
    MatrixXf item_gramian = item_embedding_.transpose() * item_embedding_;
    float loss_unobserved = unobserved_weight_ *
                            (user_gramian.array() * item_gramian.array()).sum();

    // Exposure regulariser.
    VectorXf user_embedding_avg = user_embedding_.colwise().mean();
    float reg_expo = ExpoRegularizationValue(num_users);
    float loss_reg_expo =
        reg_expo * (item_embedding_ * user_embedding_avg).squaredNorm();

    // Total loss.
    float loss = loss_observed + loss_unobserved + loss_reg + loss_reg_expo;

    VectorXf avg_scores = item_embedding_ * user_embedding_avg;

    auto time_end = std::chrono::steady_clock::now();
    uint64_t duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            time_end - time_start)
                            .count();

    LOG(INFO) << fmt::format(
        "Loss={0:.2f} Loss_observed={1:.2f} Loss_unobserved={2:.2f} "
        "Loss_reg={3:.2f} Loss_expo={4:.2f}",
        loss, loss_observed, loss_unobserved, loss_reg, loss_reg_expo);
    LOG(INFO) << fmt::format("Time={0}", duration);
  }

  // Computes the regularisation value for a user (or item). The value depends
  // on the number of observations for this user (or item) and the total number
  // of items (or users).
  const float RegularizationValue(int history_size, int num_choices) const {
    return regularization_ *
           pow(history_size + unobserved_weight_ * num_choices,
               regularization_exp_);
  }

  // Computes the weight for the exposure regulariser. The value depends
  // on the number of users for strong generalisation settings.
  const float ExpoRegularizationValue(int num_users) const {
    return regularization_expo_ * pow(num_users, 2);
  }

  // Computes the weight for the penality term of the augmented Lagrangian.
  // The value depends on the number of users for strong generalisation
  // settings.
  const float PenaltyValue(int num_users) const {
    return admm_penalty_ * pow(num_users, 2);
  }

  // Templated implementation of the U step.
  template <typename F>
  float StepU(const SpMatrix& data_by_user, F get_user_embedding_ref,
              MatrixXf& user_embedding, const MatrixXf& item_embedding,
              const VectorXf& user_embedding_avg,
              const VectorXf& dual_embedding, const MatrixXf& gramian,
              const float learning_rate, const float admm_penalty,
              const bool print_residualstats = false) {
    float residual = 0;
    MatrixXf user_embedding_prev;
    if (print_residualstats) {
      // Copying user embedding matrix for residual computation
      user_embedding_prev = user_embedding;
    }
    // Perform the parallel part of updating the user embeddings
    StepU_parallel(data_by_user, get_user_embedding_ref, item_embedding,
                   user_embedding_avg, dual_embedding, gramian, learning_rate,
                   admm_penalty);

    // Compute the non-parallel part (proximal mapping)
    Prox(user_embedding, learning_rate, admm_penalty);

    if (print_residualstats) {
      residual = (user_embedding - user_embedding_prev).norm();
    }
    return residual;
  }

  // Templated implementation of the parallel part of the U step.
  template <typename F>
  void StepU_parallel(const SpMatrix& data_by_user, F get_user_embedding_ref,
                      const MatrixXf& item_embedding,
                      const VectorXf& user_embedding_avg,
                      const VectorXf& dual_embedding, const MatrixXf& gramian,
                      const float learning_rate, const float admm_penalty) {
    int num_users = data_by_user.size();
    int num_items = item_embedding.rows();

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
          const SpVector& train_history = data_by_user_iter->second;
          ++data_by_user_iter;
          const VectorXf& user_emb = get_user_embedding_ref(u);
          m.unlock();

          assert(!train_history.empty());

          float reg = RegularizationValue(train_history.size(), num_items);
          const VectorXf& new_user_emb =
              ProjectU(num_users, train_history, user_emb, item_embedding,
                       gramian, user_embedding_avg, dual_embedding, reg,
                       unobserved_weight_, learning_rate, admm_penalty);
          // Update the user embedding. It can be overwrite here.
          m.lock();
          get_user_embedding_ref(u).noalias() = new_user_emb;
          m.unlock();
        }
      }));
    }
    // Join all threads.
    for (auto& th : threads) {
      th.join();
    }
  }

  // Proximal mapping.
  void Prox(MatrixXf& user_embeddings, const float learning_rate,
            const float admm_penalty) {
    // Compute c·11^T(U^+1(s-w)^T) by colwise mean
    int num_users = user_embeddings.rows();
    float c =
        -admm_penalty / ((admm_penalty / num_users) + (1 / learning_rate));
    const VectorXf& u_hat = (user_embeddings.colwise().mean() / num_users) * c;
    // Add the average embedding u_hat to each row of U^.
    user_embeddings.noalias() += u_hat.transpose().replicate(num_users, 1);
  }

  // Templated implementation of the V step.
  template <typename F>
  float StepV(const SpMatrix& data_by_item, F get_item_embedding_ref,
              const MatrixXf& user_embedding, const MatrixXf& item_embedding,
              const VectorXf& user_embedding_avg, const float reg_expo,
              const bool print_residualstats = false) {
    MatrixXf item_embedding_prev;
    if (print_residualstats) {
      // Copying item embedding matrix for residual computation.
      item_embedding_prev = item_embedding_;
    }

    // Pre-computation of constants (Gramian trick).
    const MatrixXf& gramian = user_embedding.transpose() * user_embedding;
    const MatrixXf& gramian_expo =
        user_embedding_avg * user_embedding_avg.transpose();

    // Used for per item regularisation.
    int num_users = user_embedding.rows();

    std::mutex m;
    auto data_by_item_iter = data_by_item.begin();  // protected by m
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back(std::thread([&] {
        while (true) {
          // Get a new item to work on.
          m.lock();
          if (data_by_item_iter == data_by_item.end()) {
            m.unlock();
            return;
          }
          int v = data_by_item_iter->first;
          const SpVector& train_history = data_by_item_iter->second;
          ++data_by_item_iter;
          m.unlock();

          assert(!train_history.empty());
          const VectorXf& item_embedding = get_item_embedding_ref(v);
          float reg = RegularizationValue(train_history.size(), num_users);
          const VectorXf& new_item_emb = ProjectV(
              train_history, item_embedding, user_embedding, user_embedding_avg,
              gramian, gramian_expo, reg, reg_expo, unobserved_weight_);
          // Update the item embedding.
          m.lock();
          get_item_embedding_ref(v).noalias() = new_item_emb;
          m.unlock();
        }
      }));
    }
    // Join all threads.
    for (auto& th : threads) {
      th.join();
    }

    float residual = 0;
    if (print_residualstats) {
      residual = (item_embedding - item_embedding_prev).norm();
    }
    return residual;
  }

  // Implementation of the s step.
  float StepAvgU(VectorXf& user_embedding_avg, const MatrixXf& user_embedding,
                 const MatrixXf& item_embedding, const VectorXf& dual_embedding,
                 const VectorXf& user_embedding_avg_true,
                 const MatrixXf& gramian, const float reg_expo,
                 const float admm_penalty,
                 const bool print_residualstats = false) {
    int embedding_size = gramian.rows();

    const VectorXf& rhs = user_embedding_avg_true + dual_embedding;

    MatrixXf matrix = reg_expo * gramian;
    for (int i = 0; i < embedding_size; ++i) {
      matrix(i, i) += admm_penalty;
    }
    Eigen::LLT<MatrixXf, Eigen::Lower> cholesky(matrix);
    assert(cholesky.info() == Eigen::Success);
    const VectorXf& user_embedding_avg_new = cholesky.solve(rhs) * admm_penalty;

    float residual = 0;
    if (print_residualstats) {
      residual = (user_embedding_avg - user_embedding_avg_new).norm();
    }
    user_embedding_avg.noalias() = user_embedding_avg_new;
    return residual;
  }

  // Implementation of the w step.
  float StepDualVar(VectorXf& dual_embedding, const MatrixXf& user_embedding,
                    const VectorXf& user_embedding_avg,
                    const VectorXf& user_embedding_avg_true,
                    const float admm_penalty,
                    const bool print_residualstats = false) {
    const VectorXf& dual_embedding_new =
        dual_embedding + user_embedding_avg_true - user_embedding_avg;

    float residual = 0;
    if (print_residualstats) {
      residual = (dual_embedding - dual_embedding_new).norm();
    }
    dual_embedding.noalias() = dual_embedding_new;
    return residual;
  }

  const MatrixXf& item_embedding() const {
    return item_embedding_;
  }

  void SetPrintTrainStats(const bool print_trainstats) override {
    print_trainstats_ = print_trainstats;
  }

  void SetPrintResidualStats(const bool print_residualstats) {
    print_residualstats_ = print_residualstats;
  }

  void SetSaveExposureDist(const bool save_exposure_dist) override {
    save_exposure_dist_ = save_exposure_dist;
  }

  void SetSaveEmbeddings(const bool save_embeddings) override {
    save_embeddings_ = save_embeddings;
  }

private:
  MatrixXf user_embedding_;
  MatrixXf item_embedding_;
  VectorXf user_embedding_avg_;
  VectorXf dual_embedding_;

  float regularization_;
  float regularization_exp_;
  float regularization_expo_;
  float learning_rate_;
  float admm_penalty_;

  int pred_iterations_;

  int embedding_dim_;
  float unobserved_weight_;

  bool print_trainstats_;
  bool print_residualstats_;
  bool save_exposure_dist_;
  bool save_embeddings_;
};

}  // namespace frecsys

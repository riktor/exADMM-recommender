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

#include <fmt/core.h>
#include <glog/logging.h>
#include "frecsys/types.hpp"

namespace frecsys {

// Evaluation results for a single user.
struct UserEvaluationResult {
  const VectorXf recall;
  const VectorXf ndcg;
  const MatrixXf exposure_uni;
  const MatrixXf exposure_logrank;
};

// Evaluation results for testing users.
struct EvaluationResult {
  const VectorXi k_list;
  const VectorXf recall;
  const VectorXf ndcg;
  const VectorXf coverage;
  const VectorXf gini_uni;
  const VectorXf gini_logrank;

  // Generate a formatted string for measure@K.
  std::string format(std::string measure_name, VectorXf measurements) const {
    assert(k_list.size() == measurements.size());

    std::stringstream ss;

    for (int i = 0; i < measurements.size(); i++) {
      int k = k_list[i];
      ss << fmt::format("{0}@{1}={2:.4f}", measure_name, k, measurements[i]);
      if (i != measurements.size() - 1) {
        ss << " ";
      }
    }
    return ss.str();
  }

  // Emit logs of ranking measures.
  void show() const {
    LOG(INFO) << format("Rec", this->recall);
    LOG(INFO) << format("NDCG", this->ndcg);
    LOG(INFO) << format("Cov", this->coverage);
    LOG(INFO) << format("Gini(uniform)", this->gini_uni);
    LOG(INFO) << format("Gini(logrank)", this->gini_logrank);
  }
};
}  // namespace frecsys

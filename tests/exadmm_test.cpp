#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <random>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fmt/core.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "frecsys/exadmm.hpp"

using exadmm_type = frecsys::EXADMMRecommender;
#define RECSYS_NAME "frecsys::EXADMMRecommender"

class EXADMMTest : public ::testing::Test {
protected:
  void SetUp() override {
    google::InstallFailureSignalHandler();

    flags_["embedding_dim"] = "8";
    flags_["unobserved_weight"] = "0.1";
    flags_["regularization"] = "0.002";
    flags_["regularization_exp"] = "1.0";
    flags_["stddev"] = "0.1";
    flags_["print_train_stats"] = "1";
    flags_["print_residual_stats"] = "0";
    flags_["learning_rate"] = "0.01";
    flags_["admm_penalty"] = "1e-1";
    flags_["exposure_reg"] = "1e-3";
    flags_["pred_iterations"] = "20";
    flags_["epochs"] = "10";

    // dataset path
    flags_["train_data"] = "tests/ml-1m/train.csv";
    flags_["test_train_data"] = "tests/ml-1m/validation_tr.csv";
    flags_["test_test_data"] = "tests/ml-1m/validation_te.csv";
  }
  std::unordered_map<std::string, std::string> flags_;
};

TEST_F(EXADMMTest, TEST_exadmm_ML1M) {
  google::InstallFailureSignalHandler();

  // Load the datasets
  frecsys::Dataset train(flags_.at("train_data"));
  frecsys::Dataset test_tr(flags_.at("test_train_data"));
  frecsys::Dataset test_te(flags_.at("test_test_data"));

  // Create the recommender.
  frecsys::EXADMMRecommender* recommender;
  recommender = new frecsys::EXADMMRecommender(
      std::atoi(flags_.at("embedding_dim").c_str()), train.max_user() + 1,
      train.max_item() + 1, std::atof(flags_.at("regularization").c_str()),
      std::atof(flags_.at("regularization_exp").c_str()),
      std::atof(flags_.at("exposure_reg").c_str()),
      std::atof(flags_.at("learning_rate").c_str()),
      std::atof(flags_.at("admm_penalty").c_str()),
      std::atof(flags_.at("unobserved_weight").c_str()),
      std::atof(flags_.at("stddev").c_str()),
      std::atoi(flags_.at("pred_iterations").c_str()));
  ((frecsys::EXADMMRecommender*)recommender)
      ->SetPrintTrainStats(std::atoi(flags_.at("print_train_stats").c_str()));
  ((frecsys::EXADMMRecommender*)recommender)
      ->SetPrintResidualStats(
          std::atoi(flags_.at("print_residual_stats").c_str()));

  // Disable output buffer to see results without delay.
  setbuf(stdout, NULL);

  // Helper for evaluation.
  auto evaluate = [&](int epoch) {
    Eigen::VectorXi k_list = Eigen::VectorXi::Zero(4);
    k_list << 10, 20, 50, 100;
    frecsys::EvaluationResult metrics =
        recommender->EvaluateDataset(k_list, test_tr, test_te.by_user());
    LOG(INFO) << "Epoch " << epoch << ":";
    metrics.show();
    EXPECT_LE(0.2, metrics.ndcg[2]);
  };

  // Train and evaluate.
  int num_epochs = std::atoi(flags_.at("epochs").c_str());
  for (int epoch = 0; epoch < num_epochs; ++epoch) {
    auto time_train_start = std::chrono::steady_clock::now();
    recommender->Train(train);
    auto time_train_end = std::chrono::steady_clock::now();

    uint64_t train_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                              time_train_end - time_train_start)
                              .count();
    LOG(INFO) << fmt::format("Epoch: {0}, Timer: Train={1}", epoch, train_time);
  }
  evaluate(num_epochs);

  delete recommender;
  return;
}

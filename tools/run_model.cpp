//
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
#include <fmt/core.h>
#include <glog/logging.h>
#include <chrono>
#include <string>
#include "CLI11/CLI11.hpp"

#include "Eigen/Core"
#include "Eigen/Dense"

#include "frecsys/fairrec.hpp"
#include "frecsys/ials.hpp"
#include "frecsys/exadmm.hpp"

template <typename F>
void evaluate(int epoch, F recommender, frecsys::Dataset& test_tr,
              frecsys::Dataset& test_te) {
  Eigen::VectorXi k_list = Eigen::VectorXi::Zero(6);
  k_list << 3, 5, 10, 20, 50, 100;
  frecsys::EvaluationResult metrics =
      recommender->EvaluateDataset(k_list, test_tr, test_te.by_user());
  LOG(INFO) << "Epoch " << epoch << ":";
  metrics.show();
}

frecsys::Recommender* get_model(const std::string model_name,
                                const int num_users, const int num_items,
                                CLI::App& app) {
  frecsys::Recommender* recommender = nullptr;
  if (model_name != "fairrec") {
    if (model_name == "ials") {
      recommender = new frecsys::IALSRecommender(
          app.get_option("--dim")->as<int>(), num_users, num_items,
          app.get_option("--l2_reg")->as<float>(),
          app.get_option("--l2_reg_exp")->as<float>(),
          app.get_option("--alpha")->as<float>(),
          app.get_option("--stdev")->as<float>());
    } else if (model_name == "exadmm") {
      recommender = new frecsys::EXADMMRecommender(
          app.get_option("--dim")->as<int>(), num_users, num_items,
          app.get_option("--l2_reg")->as<float>(),
          app.get_option("--l2_reg_exp")->as<float>(),
          app.get_option("--exposure_reg")->as<float>(),
          app.get_option("--learning_rate")->as<float>(),
          app.get_option("--admm_penalty")->as<float>(),
          app.get_option("--alpha")->as<float>(),
          app.get_option("--stdev")->as<float>(),
          app.get_option("--pred_iterations")->as<int>());
      ((frecsys::EXADMMRecommender*)recommender)
          ->SetPrintResidualStats(
              app.get_option("--print_residual_stats")->as<bool>());
    } 
    recommender->SetPrintTrainStats(
        app.get_option("--print_train_stats")->as<bool>());
    recommender->SetSaveEmbeddings(
        app.get_option("--save_embeddings")->as<bool>());
  } else {
    recommender = new frecsys::FairRecRecommender(
        app.get_option("--user_embeddings_path")->as<std::string>(),
        app.get_option("--item_embeddings_path")->as<std::string>(),
        app.get_option("--min_assignment_prop")->as<float>(),
        app.get_option("--topk_fair")->as<int>());
  }
  recommender->SetSaveExposureDist(
      app.get_option("--save_exposure_dist")->as<bool>());
  return recommender;
}

int main(int argc, char* argv[]) {
  CLI::App app{"frecsys experimentation utility"};

  // Options
  bool print_evaluation_stats = false;
  app.add_option("--print_evaluation_stats", print_evaluation_stats,
                 "Verbosity of evaluation result per epoch");

  app.add_option("-d,--dim", "Embedding dimensionality of MF models")
      ->default_val(8);

  app.add_option("-a,--alpha",
                 "Weight of norm regularisation for recovered matrix")
      ->default_val(0.1);

  app.add_option("-r,--l2_reg", "Base weight of L2 regularisation")
      ->default_val(0.002);

  app.add_option("--l2_reg_exp",
                 "Exponent of Frequency-based L2 regularisation")
      ->default_val(1.0);

  app.add_option(
         "-s,--stdev",
         "Standard deviation of normal noises for parameter initialisation")
      ->default_val(0.1);

  app.add_option("--print_train_stats", "Verbosity of training statistics")
      ->default_val(true);

  app.add_option("--print_residual_stats",
                 "Verbosity of residual statistics for exADMM")
      ->default_val(false);

  app.add_option("--save_exposure_dist", "Emit logs of exposure distribution")
      ->default_val(false);

  app.add_option("--save_embeddings", "Emit embeddings")->default_val(false);

  // Options for exADMM
  app.add_option("--exposure_reg", "Weight of exposure control regularisation")
      ->default_val(0.001);

  app.add_option("-l,--learning_rate", "Learning rate for exADMM")
      ->default_val(0.04);

  app.add_option("--admm_penalty", "Weight of penalty term in ADMM")
      ->default_val(0.01);

  app.add_option("-i,--pred_iterations",
                 "Number of update iterations in prediction phase of exADMM")
      ->default_val(50);

  int epochs = 50;
  app.add_option("-e,--epoch", epochs, "Number of epochs");

  std::map<std::string, std::string> map{
    {"ials", "ials"}, {"exadmm", "exadmm"}, {"fairrec", "fairrec"}};
  std::string model_name;
  app.add_option("-n,--model_name", model_name,
                 "Model name in [ials, exadmm, fairrec]")
      ->required()
      ->check(CLI::CheckedTransformer(map, CLI::ignore_case));

  std::string train_data;
  app.add_option("--train_data", train_data, "Path of the training data file")
      ->required()
      ->check(CLI::ExistingFile);

  std::string test_train_data;
  app.add_option("--test_train_data", test_train_data,
                 "Path of the training data file in prediction phase")
      ->required()
      ->check(CLI::ExistingFile);

  std::string test_test_data;
  app.add_option("--test_test_data", test_test_data,
                 "Path of the testing data file in prediction phase")
      ->required()
      ->check(CLI::ExistingFile);

  bool post_proc = false;
  app.add_option("--post_proc", post_proc, "Use post-processing modes");

  app.add_option("--user_embeddings_path", "Path of the user embeddings")
      ->check(CLI::ExistingFile);

  app.add_option("--item_embeddings_path", "Path of the user embeddings")
      ->check(CLI::ExistingFile);

  app.add_option("--min_assignment_prop",
                 "Proportion of minimum assignment in FairRec");

  app.add_option("--topk_fair", "Length of ranked lists for FairRec");

  CLI11_PARSE(app, argc, argv);

  // Load the datasets
  frecsys::Dataset train(train_data);
  frecsys::Dataset test_tr(test_train_data);
  frecsys::Dataset test_te(test_test_data);

  frecsys::Recommender* recommender =
      get_model(model_name, train.max_user() + 1, train.max_item() + 1, app);

  if (model_name != "fairrec") {
    // Disable output buffer to see results without delay.
    setbuf(stdout, NULL);

    // Train and evaluate.
    for (int epoch = 0; epoch < epochs; ++epoch) {
      auto time_train_start = std::chrono::steady_clock::now();
      recommender->Train(train);
      auto time_train_end = std::chrono::steady_clock::now();

      uint64_t train_time =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              time_train_end - time_train_start)
              .count();
      LOG(INFO) << fmt::format("Epoch: {0}, Timer: Train={1}", epoch,
                               train_time);
      if (print_evaluation_stats) {
        evaluate(epoch, recommender, test_tr, test_te);
      }
    }
    evaluate(epochs, recommender, test_tr, test_te);
  } else {
    evaluate(epochs, recommender, test_tr, test_te);
  }
  return 0;
}

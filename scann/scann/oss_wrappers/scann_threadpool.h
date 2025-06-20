// Copyright 2023 The Google Research Authors.
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

#ifndef SCANN_OSS_WRAPPERS_SCANN_THREADPOOL_H_
#define SCANN_OSS_WRAPPERS_SCANN_THREADPOOL_H_

#include "absl/strings/string_view.h"
#include "unsupported/Eigen/CXX11/ThreadPool"

namespace research_scann {

class ThreadPool {
 public:
  ThreadPool(absl::string_view name, int num_threads);
  void Schedule(std::function<void()> fn);
  int NumThreads() const;

 private:
  std::unique_ptr<Eigen::ThreadPool> eigen_threadpool_;
};

}  // namespace research_scann

#endif

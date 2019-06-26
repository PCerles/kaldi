// decoder/likes-matrix.h

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_DECODER_DECODABLE_TENSOR_H_
#define KALDI_DECODER_DECODABLE_TENSOR_H_

#include <vector>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "base/kaldi-common.h"
#include "itf/decodable-itf.h"

namespace kaldi {

class DecodableTensorScaled: public DecodableInterface {
 public:
  DecodableTensorScaled(const tensorflow::TTypes<float>::UnalignedConstMatrix &likes,
                        const bool blank_last_index,
                        BaseFloat scale):
    likes_(likes), blank_last_index_(blank_last_index), scale_(scale) { }

  virtual int32 NumFramesReady() const { return likes_.dimension(0); }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }

  // Note, frames are numbered from zero.
  virtual BaseFloat LogLikelihood(int32 frame, int32 index) {

    size_t actual_blank_index = likes_.dimension(1) - 1;
    if (blank_last_index_) {
        if (index == 0) {
            return std::min<float>(blank_max_cost_, 
                                   likes_(frame, actual_blank_index));
        } else {
            return scale_ * likes_(frame, index - 1);
        }
    } else {
        return scale_ * likes_(frame, index);
    }
  }

  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() const { return likes_.dimension(1); }

 private:
  const tensorflow::TTypes<float>::UnalignedConstMatrix &likes_;
  const bool blank_last_index_;
  const float blank_max_cost_ = 0.5;
  BaseFloat scale_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableTensorScaled);
};
}  // namespace kaldi

#endif  // KALDI_DECODER_DECODABLE_TENSOR_H_

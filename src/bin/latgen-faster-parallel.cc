// bin/latgen-faster-parallel.cc

// Copyright 2009-2012  Microsoft Corporation, Karel Vesely
//                2013  Johns Hopkins University (author: Daniel Povey)
//                2013  GoVIvace Inc. (author: Nagendra Goel)
//                2014  Guoguo Chen

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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "decoder/decodable-matrix.h"
#include "base/timer.h"
#include "util/kaldi-thread.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::Fst;
    using fst::StdArc;

    const char *usage =
        "Generate lattices, reading log-likelihoods as matrices, using multiple decoding threads\n"
        "Usage: latgen-faster-parallel [options] (fst-in|fsts-rspecifier) loglikes-rspecifier"
        " lattice-wspecifier [ words-wspecifier [alignments-wspecifier] ]\n";
    ParseOptions po(usage);
    bool allow_partial = false;
    BaseFloat acoustic_scale = 0.1;
    LatticeFasterDecoderConfig config;
    TaskSequencerConfig sequencer_config; // has --num-threads option

    std::string word_syms_filename;
    config.Register(&po);
    sequencer_config.Register(&po);

    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");

    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial, "If true, produce output even if end state was not reached.");

    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() > 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string fst_in_str = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        lattice_wspecifier = po.GetArg(3),
        words_wspecifier = po.GetOptArg(4),
        alignment_wspecifier = po.GetOptArg(5);

    bool determinize = config.determinize_lattice;
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
           : lattice_writer.Open(lattice_wspecifier)))
      KALDI_ERR << "Could not open table for writing lattices: "
                 << lattice_wspecifier;

    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                   << word_syms_filename;

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;
    Fst<StdArc> *decode_fst = NULL; // only used if there is a single
                                    // decoding graph.

    TaskSequencer<DecodeUtteranceLatticeNoPhoneFasterClass> sequencer(sequencer_config);

    Timer timer; 

    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) {
      SequentialBaseFloatMatrixReader loglike_reader(feature_rspecifier);
      // Input FST is just one FST, not a table of FSTs.
      decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);

      timer.Reset();
      {
        for (; !loglike_reader.Done(); loglike_reader.Next()) {
          std::string utt = loglike_reader.Key();
          Matrix<BaseFloat> *loglikes =
            new Matrix<BaseFloat>(loglike_reader.Value());
          loglike_reader.FreeCurrent();
          if (loglikes->NumRows() == 0) {
            KALDI_WARN << "Zero-length utterance: " << utt;
            num_fail++;
            delete loglikes;
            continue;
          }

          LatticeFasterDecoder *decoder = new LatticeFasterDecoder(*decode_fst,
                                                                   config);
          DecodableMatrixScaled *decodable = new DecodableMatrixScaled(*loglikes, acoustic_scale); 
          DecodeUtteranceLatticeNoPhoneFasterClass *task =
              new DecodeUtteranceLatticeNoPhoneFasterClass(
                  decoder, decodable, word_syms, utt,
                  acoustic_scale, determinize, allow_partial, &alignment_writer,
                  &words_writer, &compact_lattice_writer, &lattice_writer,
                  &tot_like, &frame_count, &num_success, &num_fail, NULL);

          sequencer.Run(task); // takes ownership of "task",
          // and will delete it when done.
        }
      }
    } else { // We have different FSTs for different utterances.
      SequentialTableReader<fst::VectorFstHolder> fst_reader(fst_in_str);
      RandomAccessBaseFloatMatrixReader loglike_reader(feature_rspecifier);
      for (; !fst_reader.Done(); fst_reader.Next()) {
        std::string utt = fst_reader.Key();
        if (!loglike_reader.HasKey(utt)) {
          KALDI_WARN << "Not decoding utterance " << utt
                     << " because no loglikes available.";
          num_fail++;
          continue;
        }
        const Matrix<BaseFloat> *loglikes =
          new Matrix<BaseFloat>(loglike_reader.Value(utt));
        if (loglikes->NumRows() == 0) {
          KALDI_WARN << "Zero-length utterance: " << utt;
          num_fail++;
          delete loglikes;
          continue;
        }
        fst::VectorFst<StdArc> *fst = 
          new fst::VectorFst<StdArc>(fst_reader.Value());
        LatticeFasterDecoder *decoder =
          new LatticeFasterDecoder(config, fst);
        DecodableMatrixScaled *decodable = new DecodableMatrixScaled(*loglikes, acoustic_scale);
        DecodeUtteranceLatticeNoPhoneFasterClass *task =
            new DecodeUtteranceLatticeNoPhoneFasterClass(
                decoder, decodable, word_syms, utt, acoustic_scale,
                determinize, allow_partial, &alignment_writer, &words_writer,
                &compact_lattice_writer, &lattice_writer, &tot_like,
                &frame_count, &num_success, &num_fail, NULL);
        sequencer.Run(task); // takes ownership of "task",
        // and will delete it when done.
      }
    }
    sequencer.Wait();

    delete decode_fst;

    double elapsed = timer.Elapsed();
    KALDI_LOG << "Decoded with " << sequencer_config.num_threads << " threads.";
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor per thread assuming 100 frames/sec is "
              << (sequencer_config.num_threads*elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count) << " over "
              << frame_count<<" frames.";

    delete word_syms;
    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

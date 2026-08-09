#!/bin/sh
# Fetch every repo the track references into code/, pinned to the exact commits
# the file/line pointers in track/ were verified against (2026-08-07).
# Safe to re-run; already-fetched repos are skipped. ~240MB on disk when done.
set -eu
cd "$(dirname "$0")"
mkdir -p code

get() {
    name=$1; url=$2; sha=$3; shift 3
    dir="code/$name"
    if [ -e "$dir/.git" ]; then
        echo "skip $name (already fetched)"
        return
    fi
    git init -q "$dir"
    git -C "$dir" remote add origin "$url"
    if [ $# -gt 0 ]; then
        # blobless sparse checkout: materialize only the listed paths
        git -C "$dir" sparse-checkout set "$@"
        git -C "$dir" fetch -q --filter=blob:none --depth 1 origin "$sha"
    else
        git -C "$dir" fetch -q --depth 1 origin "$sha"
    fi
    git -C "$dir" checkout -q FETCH_HEAD
    echo "ok   $name @ $sha"
}

# --- core: the reference-grade kernel code (stages 0-5) ---
get metal-benchmarks      https://github.com/philipturner/metal-benchmarks      dc2adc640a1588246f4471d415aa6873cb6e3499
get Metal-Puzzles         https://github.com/abeleinin/Metal-Puzzles            d631c7f4a8209a94bafb6698ae9ea3de514418ef
get metal-matmul          https://github.com/0xekez/metal-matmul                04e80810bbf7ba96ebe26ff84a346d179ee50888
get m5-gemm               https://github.com/yaroslavvb/m5-gemm                 29414bebb522ddacaa009959f2bcdad9f5b3e5cf
get metal-flash-attention https://github.com/philipturner/metal-flash-attention 8671cddc38f19a6eadb804dee6a3ca2954b8bf32
get mlx-steel-kernels     https://github.com/ml-explore/mlx                     47bbfe8fa473d6d19037a8d97f1f7d30514e4cf6 mlx/backend/metal/kernels
get llama-cpp-metal       https://github.com/ggml-org/llama.cpp                 3653e6d6d547ec763317d9ecd0ace334a7e21359 ggml/src/ggml-metal
get luminal               https://github.com/luminal-ai/luminal                 bea18ecfb01c4e454e3f2d1979f20d8e85286f64
get applegpu              https://github.com/dougallj/applegpu                  4c5bae61086b8067231120c98b4756d7696d399c
get amx                   https://github.com/corsix/amx                         483714bb051da088d08a66724b22dd08a5db3c99
get tinygrad-notes        https://github.com/mesozoic-egg/tinygrad-notes        72cd3bd80c5d79d81dde30af38f4218c1ae382bf

# --- community: the war stories (stage 6) ---
get turboquant-mlx           https://github.com/arozanov/turboquant-mlx         6e928d715595dee9f6b6cc3968baa44e1f408d28
get llamacpp-turboquant-fork https://github.com/TheTom/llama-cpp-turboquant     2f2f32f5d9517518c9e860f30131acb09840a965 ggml/src/ggml-metal docs
get turboquant-plus-llamacpp https://github.com/TheTom/turboquant_plus          ba52ad107d1fdd02bc9be8fd85308226b75c905b
get dflash-mlx               https://github.com/eauchs/mlx-dflash               dd12f356380337e65645b57d3d67e05feeefcd05
get flash-moe                https://github.com/gorroai/flash-moe               4df3af8278c4bef2e7f6b34f61e4e2596b58e93b
get mtplx                    https://github.com/youssofal/MTPLX                 ed1c8eea501689b744c13bec6a99ee2d36d26ab5
get openevolve               https://github.com/algorithmicsuperintelligence/openevolve 411fb59c886c18704caaffb611e17cf9e7d824d2
get ane-reverse-engineering  https://github.com/maderix/ANE                     d91c9845c0784dec7753048954fc6d0e8411fe29
get drawthings-mfa           https://github.com/liuliu/ccv                      4701446c87aefc9ad13a4aa88a1d915d652f88c8 lib/nnc/mfa

# drawthings-mfa ships 147MB of precompiled shader blobs (build artifacts, not
# source). Delete them; `git -C code/drawthings-mfa checkout .` restores if needed.
rm -f code/drawthings-mfa/lib/nnc/mfa/kernels/*Precompiled.inc

echo "done. start reading: track/stage-0-hardware.md"

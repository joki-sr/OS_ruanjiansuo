#include <cstdio>
#include <string>
#include "utils.h"
#include "operators.h"


int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_path>\n", argv[0]);
        return 1;
    }
    
    std::string model_path = argv[1];
    
    llama_model model;
    gpt_vocab vocab;
    int n_ctx = 512; // A default context size for loading.

    if (llama_model_load(model_path, model, vocab, n_ctx)) {
        fprintf(stdout, "Model loaded successfully.\n");
        if (model.ctx) {
             dragon_free(model.ctx);
        }
        return 0;
    } else {
        fprintf(stderr, "Failed to load model.\n");
        return 1;
    }
}

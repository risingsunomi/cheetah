#include "models/general_mha.h"

int main() {
  
  // test empty model
  torch::manual_seed(42);

  int64_t vocab_size = 30522;
  int64_t embed_dim = 256;
  int64_t hidden_dim = 1024;
  int64_t num_heads = 8;
  int64_t head_dim = 32;
  int64_t seq_len = 10;

  GeneralMHA model(
    vocab_size,
    embed_dim,
    hidden_dim,
    num_heads,
    head_dim,
    seq_len
  );
  
  model->eval();

  auto tokens = torch::randint(0, vocab_size, {1, seq_len}, torch::kInt64);
  auto output = model->forward(tokens);

  std::cout << output.sizes() << std::endl;
  std::cout << output[0][0].slice(0, 0, 10) << std::endl;
}

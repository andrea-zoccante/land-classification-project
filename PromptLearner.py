import torch
import torch.nn as nn
from transformers import CLIPModel, AutoTokenizer


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.text_model
        self.encoder = clip_model.text_model.encoder
        self.positional_embedding = clip_model.text_model.embeddings.position_embedding
        self.ln_final = clip_model.text_model.final_layer_norm
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts):
        seq_length = prompts.size(1)  # Sequence length (number of tokens)
        positions = torch.arange(seq_length, device=prompts.device).unsqueeze(0)  # Shape [1, seq_length]
        
        # Get the positional embeddings for each position
        pos_embeddings = self.positional_embedding(positions)  # Shape [1, seq_length, embedding_dim]
        
        x = prompts + pos_embeddings
        x = x.reshape(x.size(0), x.size(1), -1)  # NLD -> LND
        encoder_outputs = self.encoder(x)

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.ln_final(pooled_output)

        text_features = self.text_projection(pooled_output)

        return text_features
class SimplePromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, n_ctx=8, ctx_init=None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.device = (
            torch.device("mps") if torch.backends.mps.is_available() else
            torch.device("cuda") if torch.cuda.is_available() else
            torch.device("cpu")
        )


        self.n_cls = len(classnames)
        self.n_ctx = n_ctx
        self.ctx_dim = clip_model.text_model.final_layer_norm.weight.shape[0]
        dtype = clip_model.dtype


        ctx_vectors = torch.empty(self.n_cls, n_ctx, self.ctx_dim, dtype=dtype).to(self.device)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_prefix = " ".join(["X"] * n_ctx)

        # print(f'Initial context: "{prompt_prefix}"')
        # print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors, requires_grad=True).to(self.device)  # to be optimized

        # Prepare the prompts
        classnames = [name.replace("_", " ") for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        # Tokenize and get embeddings for class names
        tokenized_prompts = self.tokenizer(prompts, padding=True, return_tensors="pt", truncation=True, max_length=77).to(self.device)
        
        with torch.no_grad():
            input_ids = tokenized_prompts["input_ids"]
            embedding = clip_model.text_model.embeddings.token_embedding(input_ids).type(dtype)

        # Store prefix and suffix
        self.register_buffer("token_prefix", embedding[:, :1, :])  # Start of sequence token
        self.register_buffer("token_suffix", embedding[:, 1+n_ctx:, :])  # Class names and rest

        self.tokenized_prompts = tokenized_prompts

    def forward(self):
        # The context vectors
        ctx = self.ctx

        # Concatenate the prefix, context, and suffix
        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)

        return prompts


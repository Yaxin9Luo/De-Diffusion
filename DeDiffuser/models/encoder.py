import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTModel,CLIPModel,CLIPTokenizer,BertModel,BertConfig
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, clip_model_name='openai/clip-vit-base-patch16', tau_initial=1.0):
        super(MyModel, self).__init__()
        self.tau = tau_initial
        # Load the pre-trained ViT model
        self.processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224-in21k')

        self.vit = ViTModel.from_pretrained('google/vit-large-patch16-224-in21k')
        # Initialize the attention pooler
        self.attention_pooler = AttentionPooler()
        
        # Load CLIP model for tokenizing text
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)

        # Linear layer for projecting queries to text tokens
        self.linear_proj = nn.Linear(768, self.clip_tokenizer.vocab_size)
        # Define special token IDs (assuming CLIP tokenizer's convention)
        self.sos_token_id = self.clip_tokenizer.bos_token_id  # Start of Sequence Token ID
        self.eos_token_id = self.clip_tokenizer.eos_token_id  # End of Sequence Token
        # Assuming the maximum sequence length (including [SOS] and [EOS]) is 77
        self.max_length = 77

    def forward(self, images):
        # Pass images through ViT
        device = "cuda"
        vit_inputs = self.processor(images=images, return_tensors="pt")
        vit_inputs = {k: v.to(device) for k, v in vit_inputs.items()}  # Ensure all inputs are on GPU
        vit_outputs = self.vit(**vit_inputs)
        # Apply attention pooling
        pooled_outputs = self.attention_pooler(vit_outputs.last_hidden_state)

        # Project to text tokens
        text_tokens = self.linear_proj(pooled_outputs) # [75,1,49408]
        probabilities = F.gumbel_softmax(text_tokens, tau=self.tau, hard=True, dim=-1)
        # Decode text tokens to text
        predicted_token_ids = torch.argmax(probabilities, dim=-1)
        # Additional logic to handle [SOS], [EOS], and positional encodings
        # Add [SOS] token at the beginning and [EOS] at the end
        batch_size = text_tokens.size(0)
        sos_tokens = torch.full((batch_size, 1), self.sos_token_id, device=text_tokens.device) # [75,1]
        eos_tokens = torch.full((batch_size, 1), self.eos_token_id, device=text_tokens.device) # [75,1]

        # Combine [SOS], text tokens, and [EOS]
        text_tokens = torch.cat([sos_tokens, predicted_token_ids, eos_tokens], dim=1)
        # Decode text tokens to text
        decoded_texts = [self.clip_tokenizer.decode(tokens, skip_special_tokens=True) 
                        for tokens in text_tokens]
        return decoded_texts

class AttentionPooler(nn.Module):
    """ Attention Pooler class.

    Args:
        num_queries: number of queries
        hidden_dim: hidden dimension of the transformer
        nheads: number of attention heads
        num_transformer_blocks: number of transformer blocks
    Returns:
        Output tensor with shape [N,num_queries,hidden_dim]
    Usage:
        >>> model = AttentionPooler()
        >>> output = model(x)
        
    """
    def __init__(self, num_queries=75, hidden_dim=768, nheads=12, num_transformer_blocks=5):
        super(AttentionPooler, self).__init__()
        self.num_queries = num_queries
        # Initialize the queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # Transformer configuration
        transformer_config = BertConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=num_transformer_blocks,
            num_attention_heads=nheads,
            intermediate_size=hidden_dim * 4  # Typically 4 times the hidden dimension
        )

        # Transformer model
        self.transformer = BertModel(transformer_config)
    def forward(self, x):
        # `x` is the output from CoCa ViT-L model (image backbone)
        
        # Get the initial query embeddings
        query_embeds = self.query_embed.weight.unsqueeze(1).repeat(1, x.size(0), 1)

        # Concatenate [SOS] and [EOS] tokens. Assuming they are part of your input `x`
        # You need to implement how to add these tokens

        # Pass through the transformer
        transformer_output = self.transformer(inputs_embeds=query_embeds, encoder_hidden_states=x)

        # Here, you might want to process the output further depending on your needs

        return transformer_output.last_hidden_state
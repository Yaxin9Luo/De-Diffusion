import torch
import torch.nn as nn
from transformers import ViTModel,BertModel, BertConfig

class ImageBackbone(torch.nn.Module):
    """ Image Backbone class.

    Args:
        None   
    Returns:
        Output tensor with shape [N,out_features]
    Usage:
        >>> model = ImageBackbone()
        >>> output = model(x)
    """
    def __init__(self):
        super(ImageBackbone, self).__init__()
        # Load the pre-trained ViT model
        self.model = ViTModel.from_pretrained('google/vit-large-patch16-224')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        return self.model(x)

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
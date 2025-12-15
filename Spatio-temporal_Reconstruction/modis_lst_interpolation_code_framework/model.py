import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """位置编码模块，为空间位置添加编码信息"""
    
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, d_model]
        """
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """多头注意力机制模块"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1. 线性变换并重塑为多头形式
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 3. 合并多头结果
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 4. 最终线性变换
        output = self.w_o(attention_output)
        
        return output, attention_weights
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights


class FeedForward(nn.Module):
    """前馈神经网络模块"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feedforward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 多头注意力 + 残差连接 + 层归一化
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feedforward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights


class LSTTransformer(nn.Module):
    """
    用于LST时空插补的Transformer模型
    支持多种特征类型：数值、时间、地理、分类
    """
    
    def __init__(self, config):
        super(LSTTransformer, self).__init__()
        self.config = config  # 保存模型配置，便于模型保存和加载

        # 模型配置
        self.d_model = config.get('d_model', 512)
        self.n_heads = config.get('n_heads', 12)
        self.n_layers = config.get('n_layers', 8)
        self.d_ff = config.get('d_ff', 1024)
        self.dropout = config.get('dropout', 0.1)

        # 特征维度配置
        self.num_features = config.get('num_features', 0)      # 数值特征数量
        self.time_features = config.get('time_features', 0)    # 时间特征数量
        self.loc_features = config.get('loc_features', 0)      # 地理特征数量
        self.cat_features = config.get('cat_features', {})     # 分类特征配置 {name: num_classes}
        
        # 特征嵌入层
        if self.num_features > 0:
            self.num_embedding = nn.Linear(self.num_features, self.d_model // 4)
        
        if self.time_features > 0:
            self.time_embedding = nn.Linear(self.time_features, self.d_model // 4)
            
        if self.loc_features > 0:
            self.loc_embedding = nn.Linear(self.loc_features, self.d_model // 4)
        
        # 分类特征嵌入
        self.cat_embeddings = nn.ModuleDict()
        cat_embed_dim = 0
        for name, num_classes in self.cat_features.items():
            embed_dim = min(50, (num_classes + 1) // 2)  # 嵌入维度
            self.cat_embeddings[name] = nn.Embedding(num_classes + 1, embed_dim, padding_idx=0)
            cat_embed_dim += embed_dim
        
        # 计算总的嵌入维度
        embed_parts = []
        if self.num_features > 0: embed_parts.append(self.d_model // 4)
        if self.time_features > 0: embed_parts.append(self.d_model // 4)
        if self.loc_features > 0: embed_parts.append(self.d_model // 4)
        if cat_embed_dim > 0: embed_parts.append(cat_embed_dim)
        
        total_embed_dim = sum(embed_parts)
        
        # 投影到统一维度
        self.feature_projection = nn.Linear(total_embed_dim, self.d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(self.d_model)
        
        # Transformer编码器层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.d_ff, self.dropout)
            for _ in range(self.n_layers)
        ])
        
        # 输出层
        self.output_projection = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 1)
        )
        
        logger.info(f"LSTTransformer初始化完成: d_model={self.d_model}, n_layers={self.n_layers}, "
                   f"n_heads={self.n_heads}, total_embed_dim={total_embed_dim}")
    
    def forward(self, batch_data, return_attention=False):
        """
        前向传播
        
        Args:
            batch_data: 字典，包含不同类型的特征
                - 'num': [batch_size, num_features] 数值特征
                - 'time': [batch_size, time_features] 时间特征
                - 'loc': [batch_size, loc_features] 地理特征
                - 'cat': {name: [batch_size]} 分类特征字典
            return_attention: 是否返回注意力权重
        
        Returns:
            output: [batch_size, 1] LST预测值
            attention_weights: 注意力权重（如果return_attention=True）
        """
        batch_size = next(iter(batch_data.values())).size(0)
        embeddings = []
        
        # 1. 数值特征嵌入
        if 'num' in batch_data and self.num_features > 0:
            num_embed = self.num_embedding(batch_data['num'])
            embeddings.append(num_embed)
        
        # 2. 时间特征嵌入
        if 'time' in batch_data and self.time_features > 0:
            time_embed = self.time_embedding(batch_data['time'])
            embeddings.append(time_embed)
        
        # 3. 地理特征嵌入
        if 'loc' in batch_data and self.loc_features > 0:
            loc_embed = self.loc_embedding(batch_data['loc'])
            embeddings.append(loc_embed)
        
        # 4. 分类特征嵌入
        if 'cat' in batch_data and self.cat_embeddings:
            cat_embeds = []
            for name, embedding_layer in self.cat_embeddings.items():
                if name in batch_data['cat']:
                    cat_embed = embedding_layer(batch_data['cat'][name])
                    cat_embeds.append(cat_embed)
            if cat_embeds:
                cat_embed = torch.cat(cat_embeds, dim=1)
                embeddings.append(cat_embed)
        
        # 5. 拼接所有嵌入
        if not embeddings:
            raise ValueError("没有有效的输入特征")
        
        combined_embed = torch.cat(embeddings, dim=1)  # [batch_size, total_embed_dim]
        
        # 6. 投影到统一维度
        x = self.feature_projection(combined_embed)  # [batch_size, d_model]
        
        # 7. 添加位置编码 (转换为序列格式)
        x = x.unsqueeze(1)  # [batch_size, 1, d_model]
        x = x.transpose(0, 1)  # [1, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, 1, d_model]
        
        # 8. 通过Transformer编码器
        attention_weights_list = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x)
            if return_attention:
                attention_weights_list.append(attn_weights)
        
        # 9. 输出层
        x = x.squeeze(1)  # [batch_size, d_model]
        output = self.output_projection(x)  # [batch_size, 1]
        
        if return_attention:
            return output, attention_weights_list
        else:
            return output
    
    def freeze_layers(self, num_layers_to_freeze):
        """冻结前num_layers_to_freeze层的参数"""
        for i, block in enumerate(self.transformer_blocks):
            if i < num_layers_to_freeze:
                for param in block.parameters():
                    param.requires_grad = False
                logger.info(f"冻结第 {i+1} 层参数")
    
    def unfreeze_all_layers(self):
        """解冻所有层的参数"""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("解冻所有层参数")


def create_model_config(num_features, time_features, loc_features, cat_features,
                       d_model=512, n_heads=8, n_layers=4, dropout=0.1):
    """
    创建模型配置字典
    
    Args:
        num_features: 数值特征数量
        time_features: 时间特征数量
        loc_features: 地理特征数量
        cat_features: 分类特征配置字典 {name: num_classes}
        d_model: 模型维度
        n_heads: 注意力头数
        n_layers: Transformer层数
        dropout: Dropout率
    
    Returns:
        config: 模型配置字典
    """
    return {
        'num_features': num_features,
        'time_features': time_features,
        'loc_features': loc_features,
        'cat_features': cat_features,
        'd_model': d_model,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'd_ff': d_model * 4,
        'dropout': dropout
    }


# ===========================
# 独立调试入口
# ===========================
if __name__ == "__main__":
    # 测试模型创建和前向传播
    logger.info("开始测试 LSTTransformer 模型")
    
    # 模拟数据配置
    batch_size = 32
    num_features = 10  # DEM, Slope, NDVI, T2m, SSRD, STRD, RH, D2M, VPD, SMAP
    time_features = 8  # month, day, hour, doy 的正余弦编码
    loc_features = 4   # lat_sin, lat_cos, lon_sin, lon_cos
    cat_features = {'clcd': 10}  # CLCD 有10个类别
    
    # 创建模型配置
    config = create_model_config(
        num_features=num_features,
        time_features=time_features,
        loc_features=loc_features,
        cat_features=cat_features,
        d_model=512,
        n_heads=8,
        n_layers=4,
        dropout=0.1
    )
    
    # 创建模型
    model = LSTTransformer(config)
    logger.info(f"模型创建成功，参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建模拟输入数据
    batch_data = {
        'num': torch.randn(batch_size, num_features),
        'time': torch.randn(batch_size, time_features),
        'loc': torch.randn(batch_size, loc_features),
        'cat': {'clcd': torch.randint(0, 10, (batch_size,))}
    }
    
    # 前向传播测试
    with torch.no_grad():
        output = model(batch_data)
        logger.info(f"前向传播成功，输出形状: {output.shape}")
        
        # 测试返回注意力权重
        output, attention_weights = model(batch_data, return_attention=True)
        logger.info(f"注意力权重层数: {len(attention_weights)}")
        logger.info(f"第一层注意力权重形状: {attention_weights[0].shape}")
    
    # 测试冻结和解冻
    model.freeze_layers(2)
    model.unfreeze_all_layers()
    
    logger.info("LSTTransformer 模型测试完成")
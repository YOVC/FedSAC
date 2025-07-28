from .gradfl_models import ConvModel, ResNetModel, VGGModel

def get_model(model_name, **kwargs):
    """
    根据模型名称返回相应的模型实例
    
    Args:
        model_name: 模型名称，可选值为 'conv', 'resnet', 'vgg'
        **kwargs: 传递给模型构造函数的参数
        
    Returns:
        model: 模型实例
    """
    if model_name.lower() == 'conv':
        return ConvModel(**kwargs)
    elif model_name.lower() == 'resnet18':
        return ResNetModel(**kwargs)
    elif model_name.lower() == 'vgg':
        return VGGModel(**kwargs)
    else:
        raise ValueError(f"不支持的模型名称: {model_name}，请选择 'conv', 'resnet' 或 'vgg'")
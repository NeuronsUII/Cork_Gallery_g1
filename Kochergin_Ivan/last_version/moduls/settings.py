def path_for_streamlit():
    path_icon = r'.\Corn\textures\tex_icon'
    path_json = r'.\Corn\textures\tex_json'
    path_logo = r'.\Corn\logo\logo.png'    
    return path_icon, path_json, path_logo
def model_weigths():
    config_path = r'.\Corn\config\upernet_beit-large_fp16_8x1_640x640_160k_ade20k.py'
    checkpoint_path = r'.\Corn\upernet_beit-large_fp16_8x1_640x640_160k_ade20k-8fc0dd5d.pth'
    return config_path, checkpoint_path
config_52 = {
    'name': 'mf52',
    'token': 3,  # num tokens
    'embed': 128,  # embed dim
    'stem': 8,
    'bneck': {'e': 24, 'o': 12, 's': 2},  # exp out stride
    'body': [
        # stage2
        {'inp': 12, 'exp': 36, 'out': 12, 'se': None, 'stride': 1, 'heads': 2},
        # stage3
        {'inp': 12, 'exp': 72, 'out': 24, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 24, 'exp': 72, 'out': 24, 'se': None, 'stride': 1, 'heads': 2},
        # stage4
        {'inp': 24, 'exp': 144, 'out': 48, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 48, 'exp': 192, 'out': 48, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 48, 'exp': 288, 'out': 64, 'se': None, 'stride': 1, 'heads': 2},
        # stage5
        {'inp': 64, 'exp': 384, 'out': 96, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 96, 'exp': 576, 'out': 96, 'se': None, 'stride': 1, 'heads': 2},
    ],
    'fc1': 1024,  # hid_layer
    'fc2': 1000  # num_clasess
    ,
}

config_294 = {
    'name': 'mf294',
    'token': 6,  # tokens
    'embed': 192,  # embed_dim
    'stem': 16,
    # stage1
    'bneck': {'e': 32, 'o': 16, 's': 1},  # exp out stride
    'body': [
        # stage2
        {'inp': 16, 'exp': 96, 'out': 24, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 24, 'exp': 96, 'out': 24, 'se': None, 'stride': 1, 'heads': 2},
        # stage3
        {'inp': 24, 'exp': 144, 'out': 48, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 48, 'exp': 192, 'out': 48, 'se': None, 'stride': 1, 'heads': 2},
        # stage4
        {'inp': 48, 'exp': 288, 'out': 96, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 96, 'exp': 384, 'out': 96, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 96, 'exp': 576, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 128, 'exp': 768, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
        # stage5
        {'inp': 128, 'exp': 768, 'out': 192, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 192, 'exp': 1152, 'out': 192, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 192, 'exp': 1152, 'out': 192, 'se': None, 'stride': 1, 'heads': 2},
    ],
    'fc1': 1920,  # hid_layer
    'fc2': 1000  # num_clasess
    ,
}

config_508 = {
    'name': 'mf508',
    'token': 6,  # tokens and embed_dim
    'embed': 192,
    'stem': 24,
    'bneck': {'e': 48, 'o': 24, 's': 1},
    'body': [
        {'inp': 24, 'exp': 144, 'out': 40, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 40, 'exp': 120, 'out': 40, 'se': None, 'stride': 1, 'heads': 2},

        {'inp': 40, 'exp': 240, 'out': 72, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 72, 'exp': 216, 'out': 72, 'se': None, 'stride': 1, 'heads': 2},

        {'inp': 72, 'exp': 432, 'out': 128, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 128, 'exp': 512, 'out': 128, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 128, 'exp': 768, 'out': 176, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 176, 'exp': 1056, 'out': 176, 'se': None, 'stride': 1, 'heads': 2},

        {'inp': 176, 'exp': 1056, 'out': 240, 'se': None, 'stride': 2, 'heads': 2},
        {'inp': 240, 'exp': 1440, 'out': 240, 'se': None, 'stride': 1, 'heads': 2},
        {'inp': 240, 'exp': 1440, 'out': 240, 'se': None, 'stride': 1, 'heads': 2},
    ],
    'fc1': 1920,  # hid_layer
    'fc2': 1000  # num_clasess
    ,
}

config = {
    'mf52': config_52,
    'mf294': config_294,
    'mf508': config_508
}

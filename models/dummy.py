class DummyModel:
    def __init__(self):
        pass
    
    def __call__(self, prompt: str = None, image_url: str = None):
        if prompt is None:
            return 'dummy', 'NA', 'NA', 'NA'
        else:
            return f'Dummy: {prompt}', 'NA', 'NA', 'NA'
# python -m pytest --version

from unitrain import main

def test_trial():
    assert True

def _test_config(
    base='word', 
    dataset='trial',
    do_e2e=False,
    NUM_PREFIX=1,
    block_size=128,
    batch_size=8,
    ):
    main(base=base, DATASET=dataset, do_e2e=do_e2e, NUM_PREFIX=NUM_PREFIX, block_size=block_size, batch_size=batch_size, )
    
def test_char():
    _test_config(base='char', do_e2e=False)
    
def test_sub():
    _test_config(base='sub', do_e2e=False)
    
def test_word():
    _test_config(base='word', do_e2e=False)
    
def test_echar():
    _test_config(base='char', do_e2e=True)
    
def test_esub():
    _test_config(base='sub', do_e2e=True)
    
def test_eword():
    _test_config(base='word', do_e2e=True)
    
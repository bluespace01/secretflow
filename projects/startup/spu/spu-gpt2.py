from transformers import AutoTokenizer, FlaxGPT2LMHeadModel, GPT2Config

tokenizer = AutoTokenizer.from_pretrained("gpt2")
pretrained_model = FlaxGPT2LMHeadModel.from_pretrained("gpt2")

def text_generation(input_ids, params):
    config = GPT2Config()
    model = FlaxGPT2LMHeadModel(config=config)

    for _ in range(10):
        outputs = model(input_ids=input_ids, params=params)
        next_token_logits = outputs[0][0, -1, :]
        next_token = jnp.argmax(next_token_logits)
        input_ids = jnp.concatenate([input_ids, jnp.array([[next_token]])], axis=1)
    return input_ids

import jax.numpy as jnp

inputs_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='jax')
outputs_ids = text_generation(inputs_ids, pretrained_model.params)

print('-' * 65 + '\nRun on CPU:\n' + '-' * 65)
print(tokenizer.decode(outputs_ids[0], skip_special_tokens=True))
print('-' * 65)





import secretflow as sf

# In case you have a running secretflow runtime already.
sf.shutdown()

sf.init(['alice', 'bob', 'carol'], address='local')

alice, bob = sf.PYU('alice'), sf.PYU('bob')
conf = sf.utils.testing.cluster_def(['alice', 'bob', 'carol'])
conf['runtime_config']['fxp_exp_mode'] = 1
conf['runtime_config']['experimental_disable_mmul_split'] = True
spu = sf.SPU(conf)


def get_model_params():
    pretrained_model = FlaxGPT2LMHeadModel.from_pretrained("gpt2")
    return pretrained_model.params


def get_token_ids():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return tokenizer.encode('I enjoy walking with my cute dog', return_tensors='jax')


model_params = alice(get_model_params)()
input_token_ids = bob(get_token_ids)()

device = spu
model_params_, input_token_ids_ = model_params.to(device), input_token_ids.to(device)

output_token_ids = spu(text_generation)(input_token_ids_, model_params_)

outputs_ids = sf.reveal(output_token_ids)
print('-' * 65 + '\nRun on SPU:\n' + '-' * 65)
print(tokenizer.decode(outputs_ids[0], skip_special_tokens=True))
print('-' * 65)
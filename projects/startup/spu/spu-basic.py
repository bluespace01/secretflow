#  https://www.secretflow.org.cn/en/docs/secretflow/v1.6.1b0/tutorial/spu_basics

import secretflow as sf
import spu

# Check the version of your SecretFlow
print('The version of SecretFlow: {}'.format(sf.__version__))

# In case you have a running secretflow runtime already.
sf.shutdown()
sf.init(['alice', 'bob', 'carol', 'dave'], address='local')

# 
aby3_config = sf.utils.testing.cluster_def(parties=['alice', 'bob', 'carol'])
print(aby3_config)

# Create a SPU device
spu_device = sf.SPU(aby3_config)
print(spu_device.cluster_def)

# Create a SPU program and upload it
alice, dave = sf.PYU('alice'), sf.PYU('dave')

spu_io = sf.device.SPUIO(spu_device.conf, spu_device.world_size)
bank_account = [{'id': 1111, 'deposit': 1000.25}, {'id': 2222, 'deposit': 100000.25}]

meta, io_info, *shares = spu_io.make_shares(bank_account, spu.Visibility.VIS_SECRET)

print(meta)
assert len(shares) == 12
print(shares[0])

bank_account_hat = spu_io.reconstruct(shares, io_info, meta)
print(bank_account_hat)

#-------------------------------------------------------------
# setup bank_account_spu and to SPU
bank_account_pyu = sf.to(alice, bank_account)
bank_account_spu = bank_account_pyu.to(spu_device)
print(sf.reveal(bank_account_spu))

#-------------------------------------------------------------
# setup debit_amount_spu
def debit_amount():
    return 10

debit_amount_pyu = alice(debit_amount)()

debit_amount_spu = debit_amount_pyu.to(spu_device)


print(debit_amount_spu.meta)
print(debit_amount_spu.shares_name)

print(sf.reveal(debit_amount_spu))
#-------------------------------------------------------------


#-------------------------------------------------------------
# setup deduce_from_account
def deduce_from_account(bank_account, amount):
    new_bank_account = []

    for account in bank_account:
        account['deposit'] = account['deposit'] - amount
        new_bank_account.append(account)

    return new_bank_account


new_bank_account_spu = spu_device(deduce_from_account)(bank_account_spu, debit_amount_spu)

print(sf.reveal(new_bank_account_spu))

new_bank_account_pyu = new_bank_account_spu.to(dave)
print(sf.reveal(new_bank_account_pyu))
#-------------------------------------------------------------



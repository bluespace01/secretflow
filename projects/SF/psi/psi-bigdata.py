import secretflow as sf
import os,numpy as np
import pandas as pd
import datetime

if __name__ == '__main__':
    # set the working directory to the current file path
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Check the version of your SecretFlow
    print('The version of SecretFlow: {}'.format(sf.__version__))

    # In case you have a running secretflow runtime already.
    sf.shutdown()
    sf.init(['alice', 'bob', 'carol'], address='local')

    # load file of usa_county_wise.csv to pandas dataframe

    data = pd.read_csv('./dataset/credit_card_transactions.csv')
    data['uid'] = np.arange(len(data)).astype('str')
    
    # # remove the column of 'Province_State' of data
    # data = data.drop(columns=['Province_State'])
    # data = data.drop(columns=['Combined_Key'])

    data = data.drop(columns=['merchant'])
    data = data.drop(columns=['street'])
    data = data.drop(columns=['job'])

    # split the dataset into 3 parts
    os.makedirs('./dataset', exist_ok=True)
    da, db = data.sample(frac=0.3), data.sample(frac=0.3)

    da.to_csv('./dataset/alice.csv', index=False)
    db.to_csv('./dataset/bob.csv', index=False)

    # start psi computation
    alice, bob = sf.PYU('alice'), sf.PYU('bob')
    spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

    # psi computation 单键隐私求交
    input_path = {alice: './dataset/alice.csv', bob: './dataset/bob.csv'}
    output_path = {alice: './dataset/alice_psi.csv', bob: './dataset/bob_psi.csv'}

    # save current time to time_old
    pre_time = datetime.datetime.now() 
    spu.psi_csv('uid', input_path, output_path, 'alice')
    post_time = datetime.datetime.now()
    print('\n\n\npsi computation time: {}'.format(post_time - pre_time))    

    '''
    # psi computation 单键隐私求交 verification
    import pandas as pd

    df = da.join(db.set_index('uid'), on='uid', how='inner', rsuffix='_bob', sort=True)
    expected = df[da.columns].astype({'uid': 'int64'}).reset_index(drop=True)

    da_psi = pd.read_csv('.data/alice_psi.csv')
    db_psi = pd.read_csv('.data/bob_psi.csv')

    pd.testing.assert_frame_equal(da_psi, expected)
    pd.testing.assert_frame_equal(db_psi, expected)

    print(da_psi)
    '''

    '''
    # 多键隐私求交
    #----------------------------------------------------------------------
    spu.psi_csv(['uid', 'month'], input_path, output_path, 'alice')
    df = da.join(
        db.set_index(['uid', 'month']),
        on=['uid', 'month'],
        how='inner',
        rsuffix='_bob',
        sort=True,
    )
    expected = df[da.columns].astype({'uid': 'int64'}).reset_index(drop=True)

    da_psi = pd.read_csv('.data/alice_psi.csv')
    db_psi = pd.read_csv('.data/bob_psi.csv')

    pd.testing.assert_frame_equal(da_psi, expected)
    pd.testing.assert_frame_equal(db_psi, expected)


    '''

    '''
    # 三方隐私求交
    #----------------------------------------------------------------------
    carol = sf.PYU('carol')
    spu_3pc = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob', 'carol']))

    input_path = {alice: '.data/alice.csv', bob: '.data/bob.csv', carol: '.data/carol.csv'}
    output_path = {
        alice: '.data/alice_psi.csv',
        bob: '.data/bob_psi.csv',
        carol: '.data/carol_psi.csv',
    }
    spu_3pc.psi_csv(
        ['uid', 'month'], input_path, output_path, 'alice', protocol='ECDH_PSI_3PC'
    )

    keys = ['uid', 'month']
    df = da.join(db.set_index(keys), on=keys, how='inner', rsuffix='_bob', sort=True).join(
        dc.set_index(keys), on=keys, how='inner', rsuffix='_carol', sort=True
    )
    expected = df[da.columns].astype({'uid': 'int64'}).reset_index(drop=True)

    da_psi = pd.read_csv('.data/alice_psi.csv')
    db_psi = pd.read_csv('.data/bob_psi.csv')
    dc_psi = pd.read_csv('.data/carol_psi.csv')

    pd.testing.assert_frame_equal(da_psi, expected)
    pd.testing.assert_frame_equal(db_psi, expected)
    pd.testing.assert_frame_equal(dc_psi, expected)

    '''
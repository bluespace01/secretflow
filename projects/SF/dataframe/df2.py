import pandas as pd
import secretflow as sf
from secretflow.data.vertical import VDataFrame
from secretflow.utils.simulation.datasets import load_linear

sf.shutdown()
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')
# similarly for woe in heu
spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))


parts = {
    bob: (1, 11),
    alice: (11, 22),
}
vdf = load_linear(parts=parts)

label_data = vdf['y']
y = sf.reveal(label_data.partitions[alice].data).values

from secretflow.preprocessing.binning.vert_woe_binning import VertWoeBinning
from secretflow.preprocessing.binning.vert_bin_substitution import VertBinSubstitution

binning = VertWoeBinning(spu)
bin_rules = binning.binning(
    vdf,
    binning_method="chimerge",
    bin_num=4,
    bin_names={alice: ['x14'], bob: ["x5", "x7"]},
    label_name="y",
)

woe_sub = VertBinSubstitution()
vdf = woe_sub.substitution(vdf, bin_rules)

# this is for demo only, be careful with reveal
print(sf.reveal(vdf.partitions[alice].data))
print(sf.reveal(vdf.partitions[bob].data))

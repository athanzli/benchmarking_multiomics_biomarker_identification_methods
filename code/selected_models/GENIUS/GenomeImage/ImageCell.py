
# class ImageCell:
#     def __init__(self, gene, loss_val, gain_val, mut_val, exp_val, chr, methy_val ):
#         self.gene = gene
#         self.loss_val = loss_val
#         self.gain_val = gain_val
#         self.mut_val = mut_val
#         self.methy_val = methy_val
#         self.exp_val = exp_val
#         self.chr = chr
#         self.i = -1
#         self.j = -1

# NOTE for eval-bk
class ImageCell:
    def __init__(self, gene, mod1_val, mod2_val, mod3_val, mod4_val, chr):
        self.gene = gene
        self.mod1_val = mod1_val
        self.mod2_val = mod2_val
        self.mod3_val = mod3_val
        self.mod4_val = mod4_val
        self.chr = chr
        self.i = -1
        self.j = -1


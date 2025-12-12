import os
import pandas as pd
import numpy as np

def run_mofa(
    data: pd.DataFrame,
    save_model_suffix: str | None = None,
):
    df_reset    = data.reset_index().rename(columns={'index': 'sample'})
    df_reset['group'] = 'group'
    melted      = df_reset.melt(id_vars=['sample','group'], var_name='feature_full', value_name='value')
    vf = melted['feature_full'].str.split('@', n=1, expand=True)
    melted['view']    = vf[0]
    melted['feature'] = vf[1]
    data_df = melted[['sample','group','feature','view','value']].copy()
    data_df['feature'] = data_df['view'] + '@' + data_df['feature']
    
    # train
    from mofapy2.run.entry_point import entry_point
    ent = entry_point()
    ent.set_data_df(data_df, likelihoods=['gaussian' for _ in data_df['view'].unique()])
    ent.set_model_options()
    ent.set_train_options()
    ent.build()
    ent.run()

    # save
    outfile = f"./selected_models/MOFA/mofa_output_{save_model_suffix}.hdf5" if save_model_suffix else "./mofa_output.hdf5"
    if os.path.exists(outfile):
        os.remove(outfile)
    ent.save(outfile=outfile)

    # get ft score
    import mofax as mfx
    model = mfx.mofa_model(outfile)

    # scale weights
    W = model.get_weights(df=True, scale=True) # https://rdrr.io/github/bioFAM/MOFA2/man/plot_top_weights.html?utm_source=chatgpt.com  scale: logical indicating whether to scale all weights from -1 to 1 (or from 0 to 1 if abs=TRUE). Default is TRUE. "Importantly, the weights of the features within a view have relative values and they should not be interpreted in an absolute scale. Therefore, for interpretability purposes we always recommend to scale the weights with scale=TRUE."

    ######## ft_score
    fac_cols = [c for c in W.columns if str(c).startswith("Factor")]
    idx = W.index.astype(str)
    V = idx.str.split('@').str[0]
    F = idx.str.split('@').str[1]

    vals = W[fac_cols].to_numpy(float)
    abs_vals = np.abs(vals)
    argmax = abs_vals.argmax(axis=1)
    score  = abs_vals[np.arange(abs_vals.shape[0]), argmax]
    sign = np.sign(vals[np.arange(vals.shape[0]), argmax])
    best_factor = np.array(fac_cols, dtype=object)[argmax]

    ft_score = pd.DataFrame({
        'view':    np.asarray(V, dtype=str),
        'feature': np.asarray(F, dtype=str),
        'factor':  best_factor,
        'score':   score,
        'sign':    sign
    })
    ft_score['feature_full'] = ft_score['view'] + '@' + ft_score['feature']
    ft_score.set_index('feature_full', inplace=True)
    ft_score.drop(columns=['view','feature'], inplace=True)

    ########
    model.close()
    ft_score_only_score = ft_score[['score']].copy()

    return ft_score, ft_score_only_score
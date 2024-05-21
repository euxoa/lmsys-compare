# lmsys-compare

This repository provides scripts for analyzing raw data from the LMSYS [Chatbot Arena Leaderboard](https://chat.lmsys.org), the link to data usually available in the notebook linked from that page (now https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH). 

Currently we have one tool, `lmsys-compare.py`, that compares skills of models using a logistic regression framework, akin to a 1D Item Response Theory (IRT) model. You can see the model written more clearly in the file `irt1.stan`. 

The tool takes a JSON of raw data (clean_battle_*) as input, and outputs a (Polars) data frame of relative skills of the models, including posterior uncertainty of the estimates. There are also various options, most useful of which are probably _the ability to specify a languge_ with `-l` (like Swedish, English), or the possibility to restrict to code or non-code prompts only. Note that the latest data files from LMSYS have numerous other covariates, including prompt lengths and types, that are not really extracted from the JSON by the script, but which would be easy to extract and use to further refine the comparison to a specific subset of prompts. Options:

```
  -h, --help            show this help message and exit
  --nmodels NMODELS, -n NMODELS
                        Number of models with most data to include (default: all)
  --language_regex LANGUAGE_REGEX, -l LANGUAGE_REGEX
  --code                Enable code
  --no-code             Disable code
  --models MODELS, -m MODELS
                        Models to report (regex, default: '')
  --min_mse MIN_MSE, -e MIN_MSE
                        Only report models with MSE of skill below this value.
```
So calling for example `python3 lmsys-compare.py clean_battle_20240519_public.json --code -m 'gpt-4o|opus|gpt-3.5|llama-3'` should give you a table as below:

```
Input: clean_battle_20240519_public.json
shape: (8, 5)
┌────────────────────────┬───────┬─────┬───────────┬───────────┐
│ model                  ┆ count ┆ id  ┆ skill     ┆ MSE_skill │
│ ---                    ┆ ---   ┆ --- ┆ ---       ┆ ---       │
│ str                    ┆ u32   ┆ u32 ┆ f64       ┆ f64       │
╞════════════════════════╪═══════╪═════╪═══════════╪═══════════╡
│ gpt-4o-2024-05-13      ┆ 1627  ┆ 39  ┆ 2.072369  ┆ 0.061247  │
│ claude-3-opus-20240229 ┆ 8760  ┆ 10  ┆ 1.629636  ┆ 0.032876  │
│ llama-3-70b-instruct   ┆ 8769  ┆ 47  ┆ 1.093747  ┆ 0.032902  │
│ llama-3-8b-instruct    ┆ 5412  ┆ 48  ┆ 0.557968  ┆ 0.037353  │
│ gpt-3.5-turbo-0125     ┆ 3472  ┆ 30  ┆ 0.496733  ┆ 0.041498  │
│ gpt-3.5-turbo-0613     ┆ 1579  ┆ 32  ┆ 0.466514  ┆ 0.055872  │
│ gpt-3.5-turbo-1106     ┆ 721   ┆ 33  ┆ 0.278692  ┆ 0.081789  │
│ codellama-34b-instruct ┆ 281   ┆ 13  ┆ -0.421149 ┆ 0.130257  │
└────────────────────────┴───────┴─────┴───────────┴───────────┘
```

It shows that in coding, GPT-4o is now the best model (of those chosen to the comparison, and in fact of all models). Reported differences in skills are actually log-odds for winning the blind comparison, so for example GPT-4o has odds of roughly exp(2.07-1.63)≈1.55 (against 1.0) of winning Opus in a "random" coding prompt. (What users actually prompt at the Chatbot Arena is a good question.)

The IRT model is implemented in JAX and it calculates the estimation errors with a Laplace approximation. In quick experiments, the errors were even slightly (5–10%) too conservative---in general you shouldn't take them to be absolutely accurate. Under the usual gaussian approximation (which seems to hold here quite well), the 95% confidence interval for skill is the usual ±1.96*MSE_skill.

Note that the skill values are unique up to a constant, so meaningful only when compared. 


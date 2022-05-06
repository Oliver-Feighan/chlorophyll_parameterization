%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1551_chromophore_19 TDDFT with cam-b3lyp functional

0 1
Mg   25.695   50.805   26.225
C   23.636   52.014   28.903
C   27.687   49.342   28.608
C   27.690   49.893   23.739
C   23.761   52.642   24.067
N   25.797   50.912   28.589
C   24.704   51.245   29.361
C   25.009   50.707   30.841
C   26.399   50.055   30.694
C   26.737   50.097   29.238
C   27.515   50.703   31.532
C   23.858   49.839   31.445
C   23.330   50.102   32.860
C   24.289   50.769   33.921
O   24.745   51.916   33.873
O   24.544   49.957   34.972
N   27.533   49.800   26.198
C   28.192   49.272   27.257
C   29.357   48.595   26.784
C   29.314   48.673   25.335
C   28.103   49.468   25.018
C   30.273   47.764   27.611
C   30.332   47.999   24.334
O   30.254   48.027   23.109
C   31.368   46.965   24.855
N   25.759   51.321   24.233
C   26.636   50.765   23.406
C   26.399   51.222   21.931
C   25.111   52.068   22.099
C   24.864   52.011   23.540
C   27.559   51.966   21.354
C   24.021   51.585   21.030
C   22.686   50.983   21.545
N   24.012   52.050   26.396
C   23.324   52.671   25.380
C   22.118   53.263   25.950
C   22.174   53.038   27.350
C   23.360   52.308   27.525
C   21.040   53.926   25.079
C   21.520   53.247   28.609
O   20.512   53.883   28.941
C   22.386   52.500   29.664
C   22.700   53.502   30.652
O   23.631   54.291   30.595
O   21.759   53.395   31.676
C   22.006   54.330   32.767
C   25.668   50.393   35.780
C   25.631   49.635   37.025
C   24.666   49.827   37.989
C   23.731   51.024   38.003
C   24.496   48.955   39.213
C   25.826   48.524   40.010
C   26.055   46.994   39.901
C   26.776   46.385   41.108
C   27.772   45.183   40.724
C   25.784   45.931   42.227
C   26.403   45.837   43.623
C   25.608   46.546   44.822
C   26.469   47.608   45.627
C   25.825   47.857   47.043
C   26.654   48.954   44.892
C   28.043   49.356   44.450
C   28.044   50.599   43.513
C   28.157   51.871   44.398
C   27.154   52.900   43.941
C   29.613   52.298   44.672
H   28.212   48.766   29.372
H   28.169   49.479   22.849
H   23.071   53.061   23.330
H   25.075   51.662   31.363
H   26.385   49.014   31.017
H   28.165   51.333   30.925
H   27.945   49.807   31.982
H   27.128   51.470   32.203
H   24.263   48.829   31.393
H   22.951   49.794   30.841
H   22.930   49.180   33.282
H   22.498   50.799   32.764
H   30.002   46.711   27.533
H   30.249   47.967   28.682
H   31.305   47.911   27.294
H   32.285   47.487   25.129
H   31.643   46.357   23.994
H   30.943   46.380   25.671
H   26.367   50.323   21.315
H   25.357   53.101   21.850
H   28.272   52.066   22.172
H   27.428   53.009   21.065
H   27.952   51.376   20.526
H   24.377   50.949   20.219
H   23.632   52.432   20.464
H   22.223   50.287   20.845
H   21.908   51.720   21.743
H   22.918   50.411   22.444
H   21.103   53.555   24.055
H   20.988   55.014   25.101
H   20.088   53.452   25.320
H   21.839   51.707   30.174
H   21.976   55.374   32.457
H   22.953   54.149   33.277
H   21.147   54.207   33.426
H   25.608   51.448   36.046
H   26.620   50.241   35.270
H   26.329   48.816   37.201
H   22.709   50.650   38.060
H   23.700   51.474   37.011
H   23.995   51.793   38.729
H   24.071   48.018   38.854
H   23.697   49.408   39.801
H   25.632   48.810   41.044
H   26.634   49.147   39.627
H   26.479   46.774   38.922
H   25.054   46.583   40.039
H   27.385   47.203   41.493
H   27.629   44.377   41.444
H   28.792   45.562   40.783
H   27.493   44.803   39.742
H   25.484   44.926   41.932
H   25.009   46.690   42.332
H   27.450   46.139   43.619
H   26.272   44.777   43.841
H   25.361   45.767   45.544
H   24.619   46.912   44.546
H   27.469   47.222   45.825
H   24.773   47.574   47.089
H   25.844   48.865   47.459
H   26.328   47.200   47.753
H   26.182   49.773   45.435
H   26.097   48.860   43.959
H   28.497   48.576   43.839
H   28.673   49.478   45.331
H   27.147   50.589   42.894
H   28.860   50.467   42.802
H   27.934   51.614   45.434
H   26.188   52.709   44.408
H   27.048   52.944   42.857
H   27.383   53.949   44.125
H   29.845   52.331   45.736
H   29.842   53.259   44.210
H   30.303   51.500   44.401


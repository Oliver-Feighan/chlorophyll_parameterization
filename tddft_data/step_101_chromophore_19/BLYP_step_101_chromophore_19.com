%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_101_chromophore_19 TDDFT with blyp functional

0 1
Mg   25.250   51.004   26.948
C   22.943   51.938   29.339
C   27.515   50.171   29.456
C   27.832   50.574   24.594
C   23.212   52.145   24.438
N   25.229   51.181   29.104
C   24.131   51.468   29.930
C   24.420   51.333   31.459
C   25.941   50.694   31.372
C   26.267   50.707   29.895
C   26.982   51.565   32.172
C   23.440   50.434   32.338
C   23.969   50.037   33.683
C   23.377   50.718   34.956
O   22.675   51.733   34.957
O   23.598   49.974   36.055
N   27.300   50.318   27.001
C   27.985   49.951   28.152
C   29.270   49.412   27.787
C   29.311   49.391   26.405
C   28.097   50.116   25.905
C   30.298   48.913   28.763
C   30.415   48.896   25.503
O   30.430   49.123   24.318
C   31.635   48.071   26.042
N   25.580   51.416   24.847
C   26.655   51.116   24.101
C   26.472   51.603   22.639
C   24.953   51.912   22.613
C   24.547   51.792   24.004
C   27.312   52.838   22.184
C   24.130   51.069   21.723
C   24.193   51.422   20.253
N   23.500   51.908   26.862
C   22.748   52.211   25.763
C   21.376   52.496   26.231
C   21.465   52.572   27.584
C   22.746   52.099   27.938
C   20.218   52.941   25.392
C   20.700   52.811   28.788
O   19.588   53.246   28.967
C   21.683   52.509   29.971
C   21.924   53.791   30.700
O   22.284   54.809   30.165
O   21.450   53.645   31.992
C   21.447   54.939   32.713
C   23.168   50.595   37.310
C   23.685   49.878   38.593
C   24.895   49.924   39.204
C   26.119   50.635   38.688
C   25.048   49.230   40.590
C   25.749   47.880   40.562
C   26.695   47.708   41.838
C   26.646   46.256   42.398
C   26.913   45.187   41.316
C   25.347   46.153   43.179
C   25.737   46.060   44.645
C   25.199   47.307   45.474
C   26.267   47.998   46.425
C   25.719   47.678   47.855
C   26.516   49.487   46.227
C   27.868   49.858   45.628
C   28.010   51.426   45.541
C   28.131   52.003   44.052
C   29.477   52.567   43.700
C   27.128   53.114   43.675
H   28.101   49.794   30.297
H   28.579   50.449   23.807
H   22.497   52.420   23.660
H   24.248   52.362   31.773
H   25.973   49.696   31.809
H   27.898   51.634   31.585
H   27.239   51.158   33.150
H   26.601   52.578   32.302
H   23.315   49.466   31.853
H   22.474   50.938   32.352
H   24.960   50.469   33.818
H   23.993   48.953   33.794
H   30.109   49.175   29.804
H   31.256   49.393   28.559
H   30.355   47.825   28.748
H   31.202   47.327   26.711
H   32.274   48.756   26.598
H   32.227   47.432   25.386
H   26.725   50.871   21.872
H   24.760   52.938   22.299
H   27.760   53.389   23.010
H   26.801   53.481   21.467
H   28.252   52.571   21.702
H   23.059   51.043   21.924
H   24.604   50.087   21.720
H   24.012   50.476   19.742
H   25.051   52.054   20.025
H   23.292   51.983   20.004
H   20.313   52.609   24.358
H   20.192   54.025   25.501
H   19.316   52.576   25.883
H   21.188   51.844   30.679
H   22.133   55.698   32.336
H   21.557   54.802   33.788
H   20.438   55.338   32.612
H   22.098   50.797   37.289
H   23.472   51.642   37.338
H   22.816   49.706   39.229
H   26.439   51.182   39.575
H   25.836   51.293   37.866
H   26.780   49.893   38.241
H   24.054   48.992   40.967
H   25.589   49.871   41.286
H   26.397   47.796   39.690
H   24.969   47.125   40.461
H   26.441   48.361   42.673
H   27.663   48.091   41.516
H   27.474   46.166   43.101
H   26.680   45.597   40.334
H   26.423   44.221   41.441
H   27.981   44.978   41.256
H   24.932   45.202   42.847
H   24.685   46.991   42.961
H   26.804   45.908   44.806
H   25.333   45.158   45.106
H   24.249   47.016   45.922
H   24.876   48.100   44.799
H   27.176   47.431   46.224
H   26.546   47.104   48.274
H   24.834   47.045   47.922
H   25.567   48.534   48.512
H   26.450   49.967   47.203
H   25.671   49.990   45.758
H   27.871   49.291   44.697
H   28.764   49.520   46.148
H   28.839   51.696   46.195
H   27.156   51.927   45.998
H   27.877   51.295   43.263
H   29.481   53.656   43.680
H   30.005   52.055   42.896
H   30.142   52.502   44.561
H   26.141   52.942   44.105
H   26.921   53.047   42.607
H   27.380   54.133   43.972


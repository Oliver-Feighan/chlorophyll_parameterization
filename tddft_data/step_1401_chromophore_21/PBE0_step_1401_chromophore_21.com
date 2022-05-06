%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1401_chromophore_21 TDDFT with PBE1PBE functional

0 1
Mg   15.613   51.961   25.180
C   17.200   50.811   28.161
C   13.053   53.023   27.287
C   14.117   53.409   22.647
C   18.253   50.904   23.348
N   15.221   51.966   27.509
C   16.077   51.570   28.478
C   15.498   51.890   29.921
C   14.111   52.403   29.574
C   14.156   52.486   28.014
C   12.863   51.599   30.197
C   16.472   52.914   30.562
C   16.443   52.980   32.134
C   15.503   52.106   32.971
O   15.881   51.184   33.682
O   14.201   52.469   32.809
N   13.725   52.851   24.943
C   12.844   53.206   25.919
C   11.618   53.837   25.303
C   11.975   54.101   23.972
C   13.278   53.358   23.770
C   10.456   54.248   26.111
C   11.235   54.957   22.936
O   11.599   55.122   21.768
C   10.028   55.695   23.271
N   16.111   52.245   23.321
C   15.399   52.882   22.392
C   16.148   52.957   21.000
C   17.516   52.311   21.436
C   17.292   51.775   22.726
C   15.437   52.328   19.800
C   18.737   53.264   21.445
C   19.715   53.058   20.272
N   17.369   51.032   25.613
C   18.340   50.612   24.733
C   19.301   49.845   25.471
C   18.928   49.880   26.810
C   17.726   50.559   26.845
C   20.544   49.319   24.913
C   19.280   49.530   28.138
O   20.177   48.822   28.619
C   18.231   50.190   29.081
C   17.642   49.087   29.989
O   16.594   48.541   29.871
O   18.467   49.028   31.076
C   18.077   48.269   32.245
C   13.290   51.785   33.728
C   12.088   52.728   33.870
C   10.906   52.462   34.431
C   10.604   51.232   35.208
C   9.773   53.473   34.354
C   9.617   54.337   35.613
C   9.179   55.832   35.439
C   7.696   56.081   35.526
C   7.193   57.309   34.647
C   7.367   56.276   37.042
C   5.926   55.881   37.344
C   5.346   56.561   38.632
C   4.241   57.606   38.253
C   4.818   58.979   38.332
C   2.928   57.438   39.176
C   1.648   57.087   38.289
C   0.605   58.156   38.252
C   0.257   58.557   36.788
C   -0.874   59.616   36.880
C   -0.261   57.307   35.948
H   12.269   53.305   27.993
H   13.710   54.007   21.829
H   19.099   50.639   22.711
H   15.419   51.042   30.601
H   14.066   53.423   29.955
H   13.178   50.778   30.841
H   12.238   51.173   29.413
H   12.298   52.288   30.825
H   16.170   53.897   30.199
H   17.524   52.816   30.294
H   16.176   54.033   32.225
H   17.467   52.789   32.453
H   10.268   53.476   26.857
H   9.535   54.283   25.530
H   10.631   55.195   26.621
H   10.355   56.106   24.226
H   9.260   54.941   23.444
H   9.860   56.486   22.540
H   16.227   54.024   20.790
H   17.768   51.558   20.690
H   15.215   53.091   19.054
H   14.456   51.906   20.020
H   16.022   51.577   19.271
H   19.311   53.083   22.354
H   18.342   54.277   21.365
H   19.402   52.419   19.446
H   20.624   52.650   20.715
H   20.036   54.000   19.828
H   21.334   50.068   24.967
H   20.354   49.160   23.852
H   20.741   48.317   25.294
H   18.743   50.973   29.639
H   17.787   47.284   31.877
H   17.237   48.647   32.827
H   18.906   48.112   32.935
H   13.691   51.479   34.694
H   13.012   50.928   33.115
H   12.196   53.675   33.341
H   9.716   50.769   34.779
H   10.517   51.496   36.262
H   11.459   50.556   35.172
H   8.912   52.866   34.077
H   10.022   54.056   33.468
H   10.565   54.220   36.138
H   8.869   53.774   36.172
H   9.440   56.186   34.441
H   9.790   56.309   36.206
H   7.114   55.193   35.280
H   6.935   56.937   33.655
H   7.996   58.046   34.662
H   6.252   57.688   35.045
H   7.534   57.308   37.353
H   8.030   55.647   37.635
H   5.875   54.793   37.369
H   5.386   56.138   36.432
H   6.127   57.015   39.241
H   4.986   55.781   39.304
H   4.027   57.436   37.197
H   5.803   59.138   37.895
H   4.829   59.255   39.386
H   4.229   59.717   37.788
H   2.883   58.408   39.669
H   2.887   56.755   40.024
H   1.102   56.250   38.723
H   1.931   56.963   37.244
H   1.020   59.005   38.795
H   -0.311   57.783   38.711
H   1.018   59.114   36.241
H   -1.705   59.335   36.232
H   -0.631   60.614   36.515
H   -1.242   59.841   37.881
H   0.571   56.656   35.680
H   -0.821   57.575   35.052
H   -0.844   56.695   36.636

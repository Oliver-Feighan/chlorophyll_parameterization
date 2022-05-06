%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_451_chromophore_18 TDDFT with blyp functional

0 1
Mg   35.329   49.311   25.040
C   34.948   47.277   28.047
C   34.093   51.984   26.930
C   35.049   50.964   22.176
C   36.107   46.415   23.399
N   34.659   49.539   27.180
C   34.654   48.649   28.273
C   34.306   49.380   29.629
C   33.639   50.695   29.042
C   34.142   50.742   27.609
C   32.116   50.925   29.116
C   35.531   49.517   30.430
C   35.424   49.287   31.925
C   34.032   49.208   32.566
O   33.333   48.204   32.518
O   33.608   50.385   33.022
N   34.795   51.293   24.602
C   34.319   52.259   25.566
C   34.088   53.479   24.846
C   34.475   53.258   23.470
C   34.703   51.791   23.337
C   33.769   54.758   25.657
C   34.582   54.267   22.361
O   35.004   53.975   21.222
C   34.124   55.657   22.621
N   35.609   48.793   23.083
C   35.567   49.640   22.062
C   35.959   49.009   20.742
C   36.609   47.701   21.246
C   36.083   47.565   22.663
C   34.675   48.898   19.864
C   38.230   47.678   20.957
C   39.244   47.611   22.119
N   35.566   47.244   25.581
C   35.887   46.213   24.773
C   36.006   44.993   25.542
C   35.690   45.388   26.882
C   35.433   46.770   26.835
C   36.331   43.615   25.049
C   35.634   44.874   28.223
O   35.876   43.727   28.640
C   35.118   46.205   29.058
C   34.000   45.901   29.945
O   32.883   46.399   29.856
O   34.360   45.007   30.950
C   33.257   44.497   31.822
C   32.437   50.333   33.866
C   31.917   51.726   33.832
C   30.905   52.253   34.554
C   30.225   51.476   35.602
C   30.673   53.745   34.518
C   31.541   54.645   35.297
C   32.503   55.489   34.407
C   32.652   56.921   34.726
C   34.002   57.458   34.370
C   31.491   57.922   34.479
C   30.628   58.362   35.782
C   31.142   59.622   36.470
C   30.105   60.761   36.458
C   30.707   62.117   36.773
C   29.079   60.420   37.579
C   27.632   60.641   37.137
C   26.877   61.907   37.574
C   26.926   63.052   36.549
C   26.514   64.380   37.251
C   25.853   62.859   35.484
H   33.711   52.831   27.503
H   35.048   51.522   21.237
H   36.557   45.503   23.001
H   33.487   48.849   30.113
H   33.907   51.573   29.630
H   31.628   50.156   29.715
H   31.613   50.979   28.150
H   31.906   51.903   29.548
H   36.025   50.461   30.197
H   36.265   48.818   30.027
H   35.839   50.104   32.516
H   36.031   48.426   32.207
H   34.609   55.436   25.508
H   33.706   54.506   26.716
H   32.885   55.218   25.215
H   33.158   55.647   23.126
H   34.023   56.237   21.703
H   34.782   56.224   23.278
H   36.704   49.660   20.285
H   36.291   46.802   20.718
H   34.733   47.953   19.323
H   34.787   49.723   19.160
H   33.776   49.039   20.464
H   38.419   48.485   20.249
H   38.529   46.871   20.288
H   40.155   48.190   21.969
H   39.544   46.564   22.081
H   38.783   47.863   23.074
H   37.307   43.222   25.331
H   36.292   43.653   23.960
H   35.540   42.939   25.373
H   35.937   46.431   29.741
H   33.519   43.684   32.500
H   32.337   44.280   31.278
H   32.848   45.227   32.520
H   32.701   49.961   34.856
H   31.637   49.700   33.481
H   32.362   52.375   33.077
H   30.908   50.815   36.136
H   29.484   50.917   35.031
H   29.680   52.245   36.149
H   29.641   53.917   34.824
H   30.511   54.054   33.486
H   32.129   53.996   35.946
H   30.851   55.275   35.857
H   32.256   55.505   33.345
H   33.474   55.012   34.546
H   32.700   56.970   35.813
H   34.632   57.558   35.254
H   33.936   58.468   33.964
H   34.543   56.906   33.601
H   30.755   57.525   33.781
H   31.888   58.866   34.106
H   30.625   57.486   36.431
H   29.599   58.468   35.441
H   32.011   60.035   35.957
H   31.626   59.423   37.427
H   29.670   60.719   35.459
H   31.710   62.019   37.188
H   30.114   62.739   37.442
H   30.844   62.696   35.859
H   29.132   61.154   38.383
H   29.066   59.421   38.015
H   27.181   59.676   37.370
H   27.555   60.555   36.054
H   27.293   62.237   38.526
H   25.838   61.651   37.781
H   27.938   63.076   36.145
H   26.186   64.191   38.273
H   25.753   64.965   36.734
H   27.388   65.024   37.349
H   25.738   61.835   35.129
H   26.295   63.302   34.591
H   24.871   63.259   35.734


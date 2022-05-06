%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_551_chromophore_15 TDDFT with blyp functional

0 1
Mg   46.451   34.645   27.949
C   44.806   32.776   30.358
C   46.480   37.304   30.205
C   47.640   36.487   25.480
C   46.121   31.888   25.663
N   45.770   34.961   30.033
C   45.231   34.027   30.853
C   45.281   34.427   32.324
C   45.399   36.026   32.060
C   45.869   36.099   30.640
C   44.211   36.934   32.429
C   46.552   33.719   32.988
C   46.555   33.790   34.538
C   45.686   34.741   35.359
O   44.499   34.892   35.323
O   46.448   35.376   36.352
N   47.028   36.636   27.875
C   47.020   37.546   28.888
C   47.656   38.736   28.384
C   47.820   38.595   27.056
C   47.492   37.211   26.744
C   47.850   39.913   29.253
C   48.285   39.681   26.045
O   48.650   39.399   24.895
C   48.514   41.073   26.500
N   46.748   34.267   25.794
C   47.247   35.226   24.978
C   47.408   34.834   23.555
C   47.057   33.295   23.623
C   46.636   33.124   25.129
C   46.374   35.607   22.601
C   48.326   32.394   23.397
C   48.074   31.160   22.489
N   45.582   32.773   27.936
C   45.600   31.759   26.980
C   44.914   30.525   27.588
C   44.585   30.948   28.875
C   45.024   32.300   29.035
C   44.666   29.138   26.838
C   44.092   30.419   30.090
O   43.729   29.295   30.376
C   44.120   31.623   31.073
C   44.370   31.270   32.463
O   45.081   30.316   32.886
O   43.358   31.883   33.133
C   42.977   31.339   34.468
C   45.734   36.183   37.341
C   46.577   37.319   37.801
C   46.206   38.419   38.415
C   44.772   38.744   38.864
C   47.196   39.478   38.812
C   47.837   40.355   37.607
C   48.681   41.649   38.014
C   47.687   42.908   38.084
C   48.116   44.224   37.387
C   47.494   43.196   39.624
C   46.237   43.993   39.925
C   46.529   45.413   40.668
C   46.026   46.709   39.824
C   46.958   47.902   40.153
C   44.573   47.075   40.174
C   43.519   46.738   39.108
C   42.371   45.789   39.476
C   40.946   46.292   39.227
C   40.457   46.911   40.514
C   39.966   45.226   38.665
H   46.633   38.165   30.859
H   48.016   37.126   24.678
H   46.155   31.000   25.028
H   44.378   34.140   32.864
H   46.167   36.427   32.722
H   43.512   36.333   33.011
H   43.791   37.314   31.498
H   44.626   37.687   33.099
H   47.494   34.192   32.711
H   46.500   32.675   32.680
H   47.563   34.038   34.872
H   46.422   32.789   34.948
H   47.068   40.657   29.098
H   48.834   40.374   29.169
H   47.860   39.760   30.332
H   47.642   41.408   27.060
H   48.720   41.570   25.551
H   49.479   41.202   26.991
H   48.428   34.980   23.202
H   46.368   33.078   22.806
H   47.007   36.259   22.000
H   45.769   36.164   23.316
H   45.830   34.878   22.000
H   48.629   31.901   24.321
H   49.120   32.896   22.844
H   48.864   31.152   21.737
H   47.079   31.202   22.046
H   48.073   30.319   23.182
H   44.984   29.159   25.796
H   43.607   28.886   26.890
H   45.208   28.398   27.427
H   43.052   31.838   31.077
H   43.756   30.747   34.950
H   42.042   30.781   34.426
H   42.727   32.167   35.131
H   45.452   35.509   38.150
H   44.770   36.521   36.960
H   47.621   37.249   37.496
H   44.130   37.956   38.471
H   44.364   39.704   38.547
H   44.614   38.733   39.943
H   48.024   38.932   39.263
H   46.875   40.143   39.614
H   46.945   40.591   37.028
H   48.398   39.673   36.968
H   49.474   41.655   37.266
H   49.196   41.464   38.957
H   46.750   42.662   37.585
H   48.207   44.937   38.207
H   47.291   44.544   36.750
H   49.056   44.149   36.841
H   48.405   43.761   39.821
H   47.462   42.260   40.181
H   45.601   43.386   40.569
H   45.710   44.167   38.987
H   47.557   45.526   41.014
H   45.849   45.336   41.516
H   46.238   46.479   38.780
H   47.986   47.702   39.853
H   46.949   48.113   41.223
H   46.628   48.757   39.563
H   44.560   48.163   40.235
H   44.251   46.711   41.150
H   44.103   46.307   38.295
H   43.101   47.647   38.674
H   42.538   45.381   40.473
H   42.406   44.943   38.790
H   41.086   46.976   38.391
H   39.903   46.192   41.118
H   39.866   47.759   40.168
H   41.289   47.364   41.053
H   38.996   45.680   38.867
H   40.165   44.249   39.106
H   40.142   45.177   37.590


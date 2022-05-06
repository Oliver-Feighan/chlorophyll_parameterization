%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1651_chromophore_13 TDDFT with blyp functional

0 1
Mg   47.979   25.481   28.409
C   48.299   27.443   31.155
C   46.892   22.788   30.482
C   48.075   23.299   25.653
C   48.885   28.022   26.238
N   47.704   25.090   30.579
C   47.880   26.126   31.496
C   47.535   25.631   32.850
C   47.323   24.090   32.599
C   47.209   23.955   31.115
C   48.431   23.193   33.195
C   46.473   26.349   33.675
C   46.701   26.324   35.239
C   45.460   26.266   36.160
O   44.345   26.004   35.847
O   45.725   26.715   37.415
N   47.491   23.390   28.149
C   47.074   22.496   29.135
C   46.891   21.191   28.524
C   47.282   21.279   27.153
C   47.662   22.745   26.921
C   46.213   20.044   29.270
C   47.408   20.094   26.149
O   47.850   20.173   25.029
C   46.766   18.701   26.595
N   48.584   25.634   26.215
C   48.443   24.629   25.337
C   48.573   25.048   23.863
C   48.419   26.641   24.113
C   48.621   26.843   25.621
C   49.915   24.777   23.195
C   47.052   27.256   23.569
C   45.721   26.883   24.254
N   48.652   27.279   28.603
C   48.922   28.282   27.653
C   49.102   29.523   28.381
C   48.939   29.240   29.731
C   48.698   27.851   29.814
C   49.465   30.855   27.726
C   48.981   29.737   31.058
O   49.121   30.916   31.428
C   48.719   28.534   32.056
C   49.850   28.219   32.876
O   50.834   27.627   32.476
O   49.553   28.473   34.169
C   50.533   28.101   35.186
C   44.568   26.881   38.291
C   44.439   25.603   39.188
C   43.958   25.586   40.436
C   43.170   26.619   41.294
C   44.058   24.256   41.243
C   42.876   23.285   41.053
C   42.667   22.348   42.266
C   41.277   22.417   42.929
C   40.184   21.816   42.039
C   40.851   23.787   43.468
C   39.872   23.690   44.648
C   40.188   24.448   46.035
C   40.152   23.507   47.263
C   38.769   23.447   47.907
C   41.299   23.813   48.206
C   42.546   22.874   48.191
C   43.863   23.723   48.402
C   45.038   23.175   47.448
C   46.245   22.693   48.287
C   45.494   24.087   46.246
H   46.666   21.983   31.184
H   48.014   22.608   24.809
H   48.946   28.925   25.626
H   48.430   25.786   33.453
H   46.379   23.752   33.026
H   48.884   23.621   34.089
H   49.208   23.044   32.445
H   48.017   22.264   33.586
H   45.443   26.103   33.415
H   46.478   27.394   33.365
H   47.126   27.323   35.339
H   47.408   25.557   35.555
H   45.179   20.112   28.932
H   45.971   20.293   30.303
H   46.632   19.038   29.294
H   45.782   18.829   27.046
H   47.408   18.270   27.364
H   46.736   18.020   25.745
H   47.766   24.604   23.281
H   49.219   27.101   23.533
H   50.395   25.646   22.747
H   49.818   23.894   22.563
H   50.534   24.522   24.055
H   47.005   26.689   22.639
H   47.052   28.293   23.234
H   45.883   26.100   24.995
H   45.006   26.503   23.525
H   45.182   27.743   24.651
H   49.065   31.641   28.366
H   49.016   30.845   26.733
H   50.543   30.972   27.619
H   47.866   28.901   32.626
H   51.109   28.945   35.564
H   51.198   27.274   34.937
H   49.947   27.730   36.027
H   43.570   27.138   37.935
H   44.720   27.695   39.000
H   44.909   24.683   38.838
H   42.908   27.479   40.678
H   43.751   26.758   42.206
H   42.318   26.178   41.813
H   44.209   24.390   42.314
H   44.947   23.744   40.873
H   43.132   22.587   40.256
H   41.984   23.874   40.841
H   43.307   22.441   43.143
H   42.761   21.326   41.899
H   41.298   21.811   43.834
H   39.393   22.552   41.896
H   39.837   20.902   42.521
H   40.507   21.528   41.039
H   40.447   24.406   42.667
H   41.698   24.427   43.716
H   39.647   22.646   44.867
H   38.972   24.209   44.319
H   39.540   25.295   46.257
H   41.196   24.842   45.905
H   40.491   22.506   46.997
H   38.574   22.376   47.941
H   38.058   23.762   47.143
H   38.766   23.995   48.849
H   40.984   23.687   49.242
H   41.489   24.886   48.172
H   42.548   22.395   47.212
H   42.516   22.169   49.021
H   44.287   23.607   49.399
H   43.675   24.779   48.210
H   44.736   22.262   46.936
H   47.125   23.264   47.993
H   46.470   21.693   47.915
H   46.085   22.743   49.364
H   45.262   23.440   45.400
H   46.526   24.437   46.234
H   44.836   24.955   46.253


%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1301_chromophore_16 TDDFT with PBE1PBE functional

0 1
Mg   39.881   41.482   26.867
C   39.053   43.691   29.479
C   40.360   38.949   29.175
C   40.816   39.450   24.270
C   39.207   43.991   24.561
N   39.901   41.428   29.212
C   39.405   42.397   30.034
C   39.298   41.861   31.449
C   39.766   40.346   31.262
C   40.044   40.242   29.763
C   41.003   39.979   32.235
C   37.889   42.054   32.090
C   37.587   41.349   33.445
C   37.064   42.370   34.516
O   35.900   42.789   34.643
O   38.176   42.796   35.256
N   40.595   39.443   26.796
C   40.745   38.601   27.886
C   41.143   37.245   27.375
C   41.238   37.395   25.951
C   40.887   38.799   25.608
C   41.461   36.027   28.241
C   41.685   36.403   24.877
O   41.836   36.656   23.696
C   41.838   34.913   25.213
N   40.068   41.714   24.699
C   40.473   40.720   23.909
C   40.295   41.052   22.439
C   39.529   42.397   22.541
C   39.645   42.761   23.967
C   41.513   41.130   21.523
C   38.110   42.412   21.821
C   37.153   41.469   22.315
N   39.317   43.410   26.956
C   39.108   44.337   25.925
C   38.729   45.606   26.433
C   38.675   45.349   27.846
C   39.092   44.042   28.100
C   38.455   46.941   25.637
C   38.146   45.976   29.086
O   37.540   47.052   29.277
C   38.452   44.905   30.219
C   39.341   45.555   31.225
O   40.500   45.916   30.975
O   38.731   45.835   32.422
C   39.363   46.723   33.401
C   37.698   43.512   36.508
C   37.968   42.647   37.756
C   37.927   42.954   39.109
C   37.694   44.399   39.536
C   38.297   41.941   40.216
C   37.135   41.234   40.747
C   37.518   39.766   41.287
C   37.152   39.415   42.772
C   38.336   38.705   43.450
C   35.998   38.368   42.606
C   35.233   38.337   43.969
C   35.226   36.913   44.485
C   34.448   36.816   45.831
C   34.937   35.538   46.676
C   32.993   36.543   45.474
C   31.988   36.974   46.605
C   30.529   36.880   46.172
C   29.699   35.846   46.873
C   28.204   36.344   47.278
C   29.593   34.474   46.072
H   40.488   38.178   29.937
H   40.963   38.745   23.449
H   38.971   44.749   23.812
H   39.987   42.439   32.065
H   38.950   39.678   31.537
H   40.950   38.946   32.579
H   41.135   40.691   33.050
H   41.868   39.936   31.573
H   37.076   41.933   31.374
H   37.942   43.127   32.270
H   38.459   40.867   33.889
H   36.843   40.625   33.115
H   42.541   36.024   28.387
H   41.122   35.122   27.737
H   40.902   36.012   29.177
H   42.015   34.276   24.346
H   40.919   34.662   25.741
H   42.701   34.717   25.849
H   39.638   40.289   22.021
H   39.989   43.234   22.017
H   42.420   40.806   22.034
H   41.518   42.172   21.202
H   41.335   40.556   20.613
H   38.408   42.048   20.838
H   37.593   43.372   21.826
H   36.506   42.017   23.000
H   37.728   40.736   22.882
H   36.663   41.023   21.450
H   37.416   46.975   25.307
H   39.237   47.033   24.884
H   38.554   47.712   26.401
H   37.497   44.638   30.669
H   40.225   46.366   33.965
H   38.567   47.158   34.005
H   39.691   47.580   32.813
H   36.693   43.923   36.604
H   38.385   44.358   36.535
H   38.196   41.624   37.454
H   38.215   45.005   38.794
H   38.033   44.635   40.544
H   36.647   44.702   39.542
H   38.835   42.461   41.009
H   38.978   41.222   39.760
H   36.332   41.023   40.040
H   36.679   41.709   41.617
H   38.602   39.674   41.223
H   37.022   39.089   40.591
H   36.867   40.337   43.278
H   38.452   39.023   44.486
H   39.242   39.128   43.015
H   38.299   37.623   43.576
H   36.374   37.358   42.446
H   35.381   38.694   41.768
H   34.237   38.769   43.876
H   35.704   38.995   44.700
H   36.235   36.573   44.715
H   34.642   36.281   43.815
H   34.684   37.716   46.398
H   34.309   34.661   46.521
H   34.843   35.619   47.759
H   35.954   35.311   46.356
H   32.804   35.473   45.383
H   32.787   37.024   44.518
H   32.247   37.990   46.901
H   32.060   36.323   47.477
H   30.504   36.681   45.101
H   30.202   37.912   46.298
H   30.183   35.611   47.821
H   27.437   35.577   47.383
H   27.944   37.067   46.504
H   28.335   36.965   48.164
H   28.662   34.511   45.505
H   29.657   33.701   46.837
H   30.494   34.288   45.488


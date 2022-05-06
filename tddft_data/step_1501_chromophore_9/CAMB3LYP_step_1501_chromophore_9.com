%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1501_chromophore_9 TDDFT with cam-b3lyp functional

0 1
Mg   35.667   0.625   29.685
C   33.357   1.867   32.174
C   38.140   0.726   32.060
C   37.945   -0.016   27.304
C   33.223   1.360   27.272
N   35.693   1.299   31.829
C   34.628   1.531   32.699
C   35.128   1.682   34.104
C   36.704   1.464   34.022
C   36.876   1.159   32.575
C   37.604   2.570   34.560
C   34.330   0.890   35.237
C   34.020   1.650   36.467
C   35.155   2.522   36.935
O   35.193   3.700   36.691
O   36.091   1.786   37.666
N   37.717   0.507   29.658
C   38.584   0.423   30.759
C   39.919   -0.036   30.387
C   39.901   -0.272   28.991
C   38.495   0.039   28.565
C   41.025   -0.164   31.462
C   41.089   -0.660   27.994
O   40.868   -0.702   26.772
C   42.462   -0.757   28.595
N   35.596   0.699   27.604
C   36.703   0.399   26.822
C   36.367   0.617   25.309
C   34.827   0.970   25.394
C   34.494   0.955   26.852
C   37.312   1.567   24.507
C   33.823   0.316   24.421
C   33.252   -1.037   24.889
N   33.755   1.339   29.657
C   32.816   1.553   28.577
C   31.530   1.935   29.082
C   31.695   2.067   30.473
C   33.007   1.763   30.776
C   30.350   2.314   28.196
C   31.063   2.400   31.719
O   29.908   2.718   31.978
C   32.031   2.216   32.950
C   32.147   3.422   33.792
O   33.031   4.214   33.648
O   31.275   3.428   34.911
C   31.515   4.738   35.679
C   36.933   2.720   38.428
C   37.733   1.936   39.438
C   37.311   1.743   40.778
C   35.957   2.180   41.352
C   38.149   0.810   41.624
C   39.617   1.346   41.762
C   40.451   0.368   42.634
C   41.402   -0.318   41.645
C   41.328   -1.900   41.582
C   42.907   0.067   41.896
C   43.357   1.503   41.633
C   43.574   2.489   42.912
C   43.842   3.941   42.441
C   45.233   4.373   42.894
C   42.711   4.783   43.088
C   42.433   6.089   42.280
C   42.844   7.372   43.017
C   43.933   8.186   42.358
C   44.719   9.051   43.361
C   43.367   8.889   41.152
H   38.744   0.512   32.944
H   38.689   -0.310   26.561
H   32.373   1.587   26.625
H   35.000   2.738   34.341
H   36.863   0.548   34.593
H   37.257   3.262   35.328
H   38.257   3.068   33.843
H   38.186   1.887   35.179
H   34.940   0.053   35.575
H   33.416   0.521   34.771
H   33.748   1.002   37.300
H   33.187   2.318   36.247
H   41.676   0.695   31.623
H   41.677   -1.004   31.221
H   40.580   -0.373   32.435
H   42.444   -1.665   29.198
H   42.760   0.135   29.146
H   43.172   -0.888   27.778
H   36.556   -0.348   24.839
H   34.827   1.999   25.036
H   38.084   1.834   25.229
H   36.807   2.417   24.048
H   37.805   1.043   23.688
H   34.302   0.133   23.459
H   33.018   1.031   24.249
H   33.548   -1.674   24.056
H   32.176   -1.009   25.063
H   33.761   -1.308   25.814
H   30.659   2.657   27.208
H   29.798   3.007   28.831
H   29.756   1.438   27.937
H   31.604   1.340   33.437
H   31.330   5.633   35.085
H   32.561   4.643   35.970
H   30.893   4.672   36.572
H   36.420   3.493   39.001
H   37.646   3.191   37.752
H   38.736   1.729   39.065
H   35.582   1.270   41.821
H   35.244   2.526   40.604
H   36.057   3.044   42.009
H   38.130   -0.199   41.211
H   37.802   0.857   42.656
H   39.551   2.217   42.414
H   40.005   1.487   40.753
H   39.764   -0.309   43.141
H   40.871   0.942   43.461
H   41.340   0.013   40.608
H   42.114   -2.425   42.124
H   41.388   -2.294   40.568
H   40.376   -2.189   42.027
H   43.403   -0.507   41.114
H   43.200   -0.317   42.874
H   42.474   1.851   41.098
H   44.246   1.506   41.003
H   44.437   2.104   43.454
H   42.670   2.348   43.504
H   43.774   3.906   41.354
H   45.138   4.992   43.786
H   45.634   4.821   41.984
H   45.942   3.583   43.142
H   42.870   4.853   44.164
H   41.743   4.345   42.844
H   41.368   6.110   42.045
H   43.038   6.009   41.377
H   43.209   7.227   44.033
H   41.981   8.037   43.063
H   44.601   7.391   42.027
H   44.938   10.041   42.960
H   45.660   8.619   43.703
H   44.175   9.285   44.277
H   44.164   9.347   40.566
H   42.654   9.634   41.504
H   42.952   8.059   40.580

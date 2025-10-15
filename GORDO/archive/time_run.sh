source activate SIMABC 
time python gamma_simulation.py > /dev/null
time python evodyn_gamma_v3.py run --n0 100000 --bottleneck 5 --generations 300 --repeats 100 --Ub 3.1622776601683795e-05 --tau 1 --alpha 1 --beta 0.03162277660168379 > /dev/null

include("murphyk_utils.jl")
#import Pkg
#Pkg.add("MurphykUtils")
#using MurphykUtils


#=
function prior = makeCodePrior(L, A, code)
        % set code(i)  = 0 if unknown (uniform prior for ci)
        % set code(i) in {1,..,A} to use delta function for that slot
        cs = ind2subv(A*ones(1,L), 1:A^L);
        prior = ones(1, A^L);
        for i=1:A^L
            for l=1:L
             if (code(l) > 0) && (cs(i,l) ~= code(l))
                    prior(i) = 0;
             end
            end
        end
        prior = normalize(prior);
    end

    L = 2; A =3;
            code = [1,2];
        codePriorHint = [0,2];
        codePrior = makeCodePrior(L, A, codePriorHint);
  Matlab order:
    0.0000 0.0000 0.0000 0.3333 0.3333 0.3333 0.0000 0.0000 0.0000

    In julia order , expect
0.0 0.3 0.0  0.0 0.3 0.0  0.0 0.3 0.0
=#


#ind2subv_str_test()
#subv2ind_str_test()

ind2subv_test()
sub2ind_test()

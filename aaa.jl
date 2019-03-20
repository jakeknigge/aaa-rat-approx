# ------------------------------------------------------------------------------------- #
function aaa(F::Any, Z::Any, tol::Float64=1e-13, mmax::Int64=100)
# aaa   rational approximation of data F on set Z
#       r, pol, res, zer, z, f, w, errvec = aaa(F, Z, tol, mmax)
#
# input:  F = vector of data values or a function handle
#         Z = vector of sample points
#         tol = relative tolerance tol; set to 1e-13 if omitted
#         mmax = max type is (mmax-1, mmax-1); set to 100 if omitted
# output: r = AAA approximant to F (function handle)
#         pol, res, zer = vectors of poles, resiudes and zeros
#         z, f, w = vectors of support pts, function values, weights
#         errvec = vector of errors at each step
# ------------------------------------------------------------------------------------- #
    Z = collect(Z);
    M = length(Z);                             # number of sample point
    if isa(F, Function) == true; F = F(Z) end; # convert function handle to vector
    Z = Z[:]; F = F[:];                        # work with column vectors
    SF = spdiagm(0 => vec(F));                 # left scaling matrix
    J = 1:M;                                   # initializations
    z = Complex{Float64}[];
    f = Complex{Float64}[];
    C = Complex{Float64}[];
    w = [];
    errvec = []; R = mean(F);
    for m in 1:mmax                            # main loop
        ~, j = findmax(abs.(F .- R))           # select next support point
        z = [z; Z[j]]; f = [f; F[j]];          # update support points, data values
        J = setdiff(J, [j]);                   # update index vector
        C = append!(vec(C), 1 ./ (Z .- Z[j])); # next column of Cauchy matrix
        C = reshape(C, M, :);
        Sf = Diagonal(f);                      # right scaling matrix
        A = SF*C - C*Sf;                       # Loewner matrix
        ASVD = svd(A[J,:]);                    # SVD of Loewner matrix
        w = ASVD.V[:,m];                       # weights = min. sing. vector
        N = C*(w .* f); D = C*w;               # numerator and denominator
        R = copy(F); R[J] = N[J] ./ D[J];      # rational approximation
        err = norm(F - R, Inf);                # max error at sample points
        errvec = [errvec; err];
        if err <= tol*norm(F, Inf); break end  # stop if converged
    end # main loop
    reval(zz) = rhandle(zz, z, f, w);          # AAA approximant as function handle
    pol, res, zer = prz(reval, z, f, w);       # poles, residues and zeros
    # remove Frois. doublets
    reval, pol, res, zer, z, f, w = cleanup(reval, pol, res, zer, z, f, w, Z, F);
    return reval, pol, res, zer, z, f, w, errvec
end # aaa function

# ------------------------------------------------------------------------------------- #
function prz(r, z, f, w) # compute poles, residues and zeros
    m = length(w); B = Matrix(I,m+1,m+1); B[1,1] = 0;
    E = [0 transpose(w); ones(m,1) Diagonal(z)];
    EEB = eigen(E, B);
    pol = EEB.values[.!isnan.(EEB.values)];               # poles
    dz = transpose(1e-5 .* exp.(2im .* pi .* (1:4) ./ 4));
    res = r(broadcast(+, pol, dz)) * transpose(dz) ./ 4;  # residues
    E = [0 (w.*f)'; ones(m,1) Diagonal(z)];
    EEB = eigen(E, B);
    zer = EEB.values[.!isnan.(EEB.values)];               # zeros
    return pol, res, zer
end # prz function

# ------------------------------------------------------------------------------------- #
function rhandle(zz, z, f, w)                 # evaluate r at zz
    zv = zz[:];                               # vectorize zz (if necessary)
    CC = 1 ./ broadcast(-, zv, transpose(z)); # Cauchy matrix
    r = (CC*(w .* f)) ./ (CC*w);              # AAA approx as vector
    ii = vec(1:length(r))[isnan.(r)];         # find NaN values (if any)
    for j = 1:length(ii)
        r[ii[j]] = f[vec(zv[ii[j]] .== z)][1];# force interpolation
    end # for loop
    r = reshape(r, size(zz));                 # AAA approximation
    return r
end # rhandle

# ------------------------------------------------------------------------------------- #
function cleanup(r, pol, res, zer, z, f, w, Z, F)
    m = length(z); M = length(Z);
    ii = vec(1:length(res))[abs.(res) .< 1e-13];      # find negligible residues
    ni = length(ii);
    if ni == 0; return r, pol, res, zer, z, f, w; end;
    print(ni, " Froissart doublet(s) present.\n")
    for j in 1:ni
        azp = abs.(z .- pol[ii[j]]);
        jj = findmin(azp)[2];
        splice!(z, jj, []); splice!(f, jj, []);       # remove nearest support points
    end # for loop
    for j in 1:length(z)
        idx = vec(1:length(F))[Z .== z[j]];
        splice!(F, idx[1], []);
        splice!(Z, idx[1], []);
    end # for loop
    m = m - length(ii);
    SF = spdiagm(0 => vec(F[1:M-m]))
    Sf = Diagonal(f);
    C = 1 ./ broadcast(-, Z, transpose(z));
    A = SF*C - C*Sf;
    ASVD = svd(A); w = ASVD.V[:,m];                   # solve least squares problem again
    reval(zz) = rhandle(zz, z, f, w);
    pol, res, zer = prz(reval, z, f, w);              # poles, residues and zeros
    return reval, pol, res, zer, z, f, w
end # cleanup

# ------------------------------------------------------------------------------------- #
# Example 1
X = LinRange(-1,1,1000);
G(x) = sign.(x);
r, pol, res, zer, z, f, w, errvec = aaa(G, X);
plot(X, G(X), label = "F(x) = sign(x)", title = "function vs. approximation")
plot!(X, real(r(X)), linestyle = :dash, label = "R(x) = approx.",
        legend = :bottomright)
png("e1app")
scatter(pol, label = "poles", legend = :topleft, title = "poles and unit circle")
plot!(Z, label = "unit circle", aspect_ratio = :equal)
png("e1pol")
plot(errvec, label = "error", yscale = :log10, ylabel = "norm(F - R, Inf)",
     title = "approximation error", xlabel = "n: type(n, n) approx.")
png("e1err")
# ------------------------------------------------------------------------------------- #
# Example 2
X = LinRange(-1,1,1000);
G(x) = 1 ./ (2 .+ cos.(20x .+ 1));
r, pol, res, zer, z, f, w, errvec = aaa(G, X);
plot(X, G(X), label = "F(x) = 1 / (2 + cos(20x + 1))",
     title = "function vs. approximation")
plot!(X, real(r(X)), linestyle = :dash, label = "R(x) = approx.")
png("e2app")
scatter(pol, label = "poles", legend = :topleft, title = "poles and unit circle")
plot!(Z, label = "unit circle", aspect_ratio = :equal)
png("e2pol")
plot(errvec, label = "error", yscale = :log10, ylabel = "norm(F - R, Inf)",
     title = "approximation error", xlabel = "n: type(n, n) approx.")
png("e2err")
# ------------------------------------------------------------------------------------- #
# Example 3
Z = exp.(LinRange(0,2im*pi,1000));
G(z) = log.(1.44 .- z .^2);
r, pol, res, zer, z, f, w, errvec = aaa(G, Z);
plot(real(Z), real(G(Z)), label = "F(z) = log(1.44 - z^2)",
     title = "function vs. approximation (real part)")
plot!(real(Z), real(r(Z)), linestyle = :dash, label = "R(x) = approx.")
png("e3appre")
plot(imag(Z), imag(G(Z)), label = "F(z) = log(1.44 - z^2)",
     title = "function vs. approximation (complex part)")
plot!(imag(Z), imag(r(Z)), linestyle = :dash, label = "R(x) = approx.")
png("e3appim")
scatter(pol, label = "poles", legend = :topleft, title = "poles and unit circle")
plot!(Z, label = "unit circle")
png("e3pol")
plot(errvec, label = "error", yscale = :log10, ylabel = "norm(F - R, Inf)",
     title = "approximation error", xlabel = "n: type(n, n) approx.")
png("e3err")
# ------------------------------------------------------------------------------------- #
# Example 4
Z = exp.(LinRange(0,2im*pi,1000));
G(z) = log.(2 .+ z .^ 4) ./ (1 .- 16 .* z .^ 4);
r, pol, res, zer, z, f, w, errvec = aaa(G, Z);
plot(real(Z), real(G(Z)), label = "F(z) = log(2 + z^4) / (1 - 16*z^4)",
     title = "function vs. approximation (real part)")
plot!(real(Z), real(r(Z)), linestyle = :dash, label = "R(x) = approx.")
png("e4appre")
plot(imag(Z), imag(G(Z)), label = "F(z) = log(2 + z^4) / (1 - 16*z^4)",
     title = "function vs. approximation (complex part)")
plot!(imag(Z), imag(r(Z)), linestyle = :dash, label = "R(x) = approx.")
png("e4appim")
scatter(pol, label = "poles", legend = :topleft, title = "poles and unit circle")
plot!(Z, label = "unit circle", aspect_ratio = :equal)
png("e4pol")
plot(errvec, label = "error", yscale = :log10, ylabel = "norm(F - R, Inf)",
     title = "approximation error", xlabel = "n: type(n, n) approx.")
png("e4err")

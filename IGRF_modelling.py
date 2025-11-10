import numpy as np
import warnings
# from scipy.interpolate import BSpline, PPoly

def design_gauss(radius, theta, phi, nmax, *, nmin=None, mmax=None,
                 source=None):
    """
    Computes matrices to connect the radial, colatitude and azimuthal field
    components to the magnetic potential field in terms of spherical harmonic
    coefficients (Schmidt quasi-normalized).

    Parameters
    ----------

    radius : ndarray, shape (...)
        Array containing the radius in kilometers.
    theta : ndarray, shape (...)
        Array containing the colatitude in degrees
        :math:`[0^\\circ,180^\\circ]`.
    phi : ndarray, shape (...)
        Array containing the longitude in degrees.
    nmax : int, positive
        Maximum degree of the sphercial harmonic expansion.
    nmin : int, positive, optional
        Minimum degree of the sphercial harmonic expansion (defaults to 1).
    mmax : int, positive, optional
        Maximum order of the spherical harmonic expansion (defaults to
        ``nmax``). For ``mmax = 0``, for example, only the zonal terms are
        returned.
    source : {'internal', 'external'}, optional
        Magnetic field source (default is an internal source).

    Returns
    -------
    A_radius, A_theta, A_phi : ndarray, shape (..., M)
        Matrices for radial, colatitude and azimuthal field components. The
        second dimension ``M`` varies depending on ``nmax``, ``nmin`` and
        ``mmax``.

    Warnings
    --------
    The function can also return the design matrix for the field components at
    the geographic poles, i.e. where ``theta == 0.`` or ``theta == 180.``.
    However, users should be careful when doing this because the vector basis
    for spherical geocentric coordinates,
    :math:`{{\\mathbf{e}_r, \\mathbf{e}_\\theta, \\mathbf{e}_\\phi}}`,
    depends on longitude, which is not well defined at the poles. That is,
    at the poles, any value for the longitude maps to the same location in
    euclidean coordinates but gives a different vector basis in spherical
    geocentric coordinates. Nonetheless, by choosing a specific value for the
    longitude at the poles, users can define the vector basis, which then
    establishes the meaning of the spherical geocentric components. The vector
    basis for the horizontal components is defined as

    .. math::

        \\mathbf{e}_\\theta &= \\cos\\theta\\cos\\phi\\mathbf{e}_x -
            \\cos\\theta\\sin\\phi\\mathbf{e}_y - \\sin\\theta\\mathbf{e}_z\\\\
        \\mathbf{e}_\\phi &= -\\sin\\phi\\mathbf{e}_x +
            \\cos\\phi\\mathbf{e}_y

    Hence, at the geographic north pole as given by ``theta = 0.`` and
    ``phi = 0.`` (chosen by the user), the returned design matrix ``A_theta``
    will be for components along the direction
    :math:`\\mathbf{e}_\\theta = \\mathbf{e}_x` and ``A_phi`` for components
    along :math:`\\mathbf{e}_\\phi = \\mathbf{e}_y`. However,
    if ``phi = 180.`` is chosen, ``A_theta`` will be for components along
    :math:`\\mathbf{e}_\\theta = -\\mathbf{e}_x` and ``A_phi``
    along :math:`\\mathbf{e}_\\phi = -\\mathbf{e}_y`.

    Examples
    --------
    Create the design matrices given 4 locations on the Earth's surface:

    >>> r = 6371.2
    >>> theta = np.array([90., 100., 110., 120.])
    >>> phi = 0.
    >>> A_radius, A_theta, A_phi = design_gauss(r, theta, phi, nmax=1)
    >>> A_radius
    array([[ 1.22464680e-16, 2.00000000e+00, 0.00000000e+00],
           [-3.47296355e-01, 1.96961551e+00, 0.00000000e+00],
           [-6.84040287e-01, 1.87938524e+00, 0.00000000e+00],
           [-1.00000000e+00, 1.73205081e+00, 0.00000000e+00]])

    Say, only the axial dipole coefficient is non-zero, what is the magnetic
    field in terms of spherical geocentric components?

    >>> m = np.array([-30000, 0., 0.])  # model coefficients in nT
    >>> Br = A_radius @ m; Br
    array([-3.67394040e-12, 1.04188907e+04, 2.05212086e+04, 3.00000000e+04])
    >>> Bt = A_theta @ m; Bt
    array([-30000. , -29544.23259037, -28190.77862358, -25980.76211353])
    >>> Bp = A_phi @ m; Bp
    array([0., 0., 0., 0.])

    A more complete example is given below:

    .. code-block:: python

        import numpy as np
        from chaosmagpy.model_utils import design_gauss, synth_values

        nmax = 5  # desired truncation degree, i.e. 35 model coefficients
        coeffs = np.arange(35)  # example model coefficients

        # example locations
        N = 10
        radius = 6371.2  # Earth's surface
        phi = np.linspace(-180., 180., num=N)  # azimuth in degrees
        theta = np.linspace(1., 179., num=N)  # colatitude in degrees

        # compute design matrices to compute predictions of the
        # field components from the model coefficients, each is of shape N x 35
        A_radius, A_theta, A_phi = design_gauss(r, theta, phi, nmax=nmax)

        # magnetic components computed from the model
        Br_pred = A_radius @ coeffs
        Bt_pred = A_theta @ coeffs
        Bp_pred = A_phi @ coeffs

        # compute the magnetic components directly from the coefficients
        Br, Bt, Bp = synth_values(coeffs, radius, theta, phi)
        np.linalg.norm(Br - Br_pred)  # approx 0.
        np.linalg.norm(Bt - Bt_pred)  # approx 0.
        np.linalg.norm(Bp - Bp_pred)  # approx 0.

    """

    # ensure ndarray inputs
    radius = np.asarray(radius, dtype=float)/ 6371.2
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # get shape of broadcasted result
    try:
        b = np.broadcast(radius, theta, phi)

    except ValueError:
        print('Cannot broadcast grid shapes:')
        print(f'radius: {radius.shape}')
        print(f'theta:  {theta.shape}')
        print(f'phi:    {phi.shape}')
        raise

    shape = b.shape

    theta_min = np.amin(theta)
    theta_max = np.amax(theta)

    # check if poles are included
    if (theta_min <= 0.0) or (theta_max >= 180.0):
        if (theta_min == 0.0) or (theta_max == 180.0):
            warnings.warn('Input coordinates include the poles.')
            poles = True
        else:
            raise ValueError('Colatitude outside bounds [0, 180].')
    else:
        poles = False

    # set internal source as default
    if source is None:
        source = 'internal'

    assert nmax > 0, '"nmax" must be greater than zero.'

    nmin = 1 if nmin is None else int(nmin)
    assert nmin <= nmax, '"nmin" must be smaller than "nmax".'
    assert nmin > 0, '"nmin" must be greater than zero.'

    mmax = nmax if mmax is None else int(mmax)
    assert mmax <= nmax, '"mmax" must be smaller than "nmax".'
    assert mmax >= 0, '"mmax" must be greater than or equal to zero.'

    # initialize radial dependence given the source
    if source == 'internal':
        r_n = radius**(-(nmin+2))
    elif source == 'external':
        r_n = radius**(nmin-1)
    else:
        raise ValueError("Source must be either 'internal' or 'external'.")

    # compute associated Legendre polynomials as (n, m, theta-points)-array
    Pnm = legendre_poly(nmax, theta)
    sinth = Pnm[1, 1]

    phi = np.radians(phi)

    # find the poles
    if poles:
        where_poles = (theta == 0.) | (theta == 180.)

    # compute the number of dimensions based on nmax, nmin, mmax
    if mmax >= (nmin-1):
        dim = int(mmax*(mmax+2) + (nmax-mmax)*(2*mmax+1) - nmin**2 + 1)
    else:
        dim = int((nmax-nmin+1)*(2*mmax+1))

    # allocate A_radius, A_theta, A_phi in memeory
    A_radius = np.zeros(shape + (dim,))
    A_theta = np.zeros(shape + (dim,))
    A_phi = np.zeros(shape + (dim,))

    num = 0
    for n in range(nmin, nmax+1):

        if source == 'internal':
            A_radius[..., num] = (n+1) * Pnm[n, 0] * r_n
        else:
            A_radius[..., num] = -n * Pnm[n, 0] * r_n

        A_theta[..., num] = -Pnm[0, n+1] * r_n

        num += 1
        for m in range(1, min(n, mmax)+1):

            cmp = np.cos(m*phi)
            smp = np.sin(m*phi)

            if source == 'internal':
                A_radius[..., num] = (n+1) * Pnm[n, m] * r_n * cmp
                A_radius[..., num+1] = (n+1) * Pnm[n, m] * r_n * smp
            else:
                A_radius[..., num] = -n * Pnm[n, m] * r_n * cmp
                A_radius[..., num+1] = -n * Pnm[n, m] * r_n * smp

            A_theta[..., num] = -Pnm[m, n+1] * r_n * cmp
            A_theta[..., num+1] = -Pnm[m, n+1] * r_n * smp

            # need special treatment at the poles because Pnm/sinth = 0/0 for
            # theta in {0., 180.},
            # use L'Hopital's rule to take the limit:
            # lim Pnm/sinth = k*dPnm | theta in {0., 180}
            # (evaluated at poles, where k=1 for theta=0 and k=-1 for
            # theta=180.); k = costh = P[1, 0] at poles for convenience
            if poles:
                div_Pnm = np.where(where_poles, Pnm[m, n+1], Pnm[n, m])
                div_Pnm[where_poles] *= Pnm[1, 0][where_poles]
                div_Pnm[~where_poles] /= sinth[~where_poles]
            else:
                div_Pnm = Pnm[n, m] / sinth

            A_phi[..., num] = m * div_Pnm * r_n * smp
            A_phi[..., num+1] = -m * div_Pnm * r_n * cmp

            num += 2

        # update radial dependence given the source
        if source == 'internal':
            r_n = r_n / radius
        else:
            r_n = r_n * radius

    return A_radius, A_theta, A_phi


def legendre_poly(nmax, theta):
    """
    Returns associated Legendre polynomials :math:`P_n^m(\\cos\\theta)`
    (Schmidt quasi-normalized) and the derivative
    :math:`dP_n^m(\\cos\\theta)/d\\theta` evaluated at :math:`\\theta`.

    Parameters
    ----------
    nmax : int, positive
        Maximum degree of the spherical expansion.
    theta : ndarray, shape (...)
        Colatitude in degrees :math:`[0^\\circ, 180^\\circ]`
        of arbitrary shape.

    Returns
    -------
    Pnm : ndarray, shape (n, m, ...)
          Evaluated values and derivatives, grid shape is appended as trailing
          dimensions. :math:`P_n^m(\\cos\\theta)` := ``Pnm[n, m, ...]`` and
          :math:`dP_n^m(\\cos\\theta)/d\\theta` := ``Pnm[m, n+1, ...]``

    References
    ----------
    Based on Equations 26-29 and Table 2 in:

    Langel, R. A., "Geomagnetism - The main field", Academic Press, 1987,
    chapter 4

    """

    costh = np.cos(np.radians(theta))
    sinth = np.sqrt(1-costh**2)

    Pnm = np.zeros((nmax+1, nmax+2) + costh.shape)
    Pnm[0, 0] = 1.  # is copied into trailing dimensions
    Pnm[1, 1] = sinth  # write theta into trailing dimenions via broadcasting

    rootn = np.sqrt(np.arange(2 * nmax**2 + 1))

    # Recursion relations after Langel "The Main Field" (1987),
    # eq. (27) and Table 2 (p. 256)
    for m in range(nmax):
        Pnm_tmp = rootn[m+m+1] * Pnm[m, m]
        Pnm[m+1, m] = costh * Pnm_tmp

        if m > 0:
            Pnm[m+1, m+1] = sinth*Pnm_tmp / rootn[m+m+2]

        for n in range(m+2, nmax+1):
            d = n * n - m * m
            e = n + n - 1
            Pnm[n, m] = ((e * costh * Pnm[n-1, m] - rootn[d-e] * Pnm[n-2, m])
                         / rootn[d])

    # dP(n,m) = Pnm(m,n+1) is the derivative of P(n,m) vrt. theta
    Pnm[0, 2] = -Pnm[1, 1]
    Pnm[1, 2] = Pnm[1, 0]
    for n in range(2, nmax+1):
        Pnm[0, n+1] = -np.sqrt((n*n + n) / 2) * Pnm[n, 1]
        Pnm[1, n+1] = ((np.sqrt(2 * (n*n + n)) * Pnm[n, 0]
                       - np.sqrt((n*n + n - 2)) * Pnm[n, 2]) / 2)

        for m in range(2, n):
            Pnm[m, n+1] = (0.5*(np.sqrt((n + m) * (n - m + 1)) * Pnm[n, m-1]
                           - np.sqrt((n + m + 1) * (n - m)) * Pnm[n, m+1]))

        Pnm[n, n+1] = np.sqrt(2 * n) * Pnm[n, n-1] / 2

    return Pnm

r = np.array([6371.2, 6371.2, 6371.2, 6371.2])
theta = np.array([90., 100., 110., 120.])
phi = np.array([0., 0., 0., 0.])
A_radius, A_theta, A_phi = design_gauss(r, theta, phi, nmax=1)
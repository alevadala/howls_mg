program TwoPointCorrelation
    implicit none
    
    ! Libraries
    external besselj0
    external quadgk
    external readdlm

    ! Constants
    real, parameter :: pi = 3.14159265358979323846
    real, parameter :: theta_min = 0.01
    real, parameter :: theta_max = 100.0
    integer, parameter :: num_theta = 100
    integer, parameter :: num_ell = 100
    
    ! Arrays
    real :: theta(num_theta), xi_plus(num_theta)
    real :: ell(num_ell), P_kappa(num_ell)
    
    ! Load matter power spectrum from file
    call readdlm("\home\alessandro\phd\scripts\Dustgrain_outs\Cls\fr4_0.5")
    
    ! Define the range of ell values
    ell = [(10.0 + (100000.0 - 10.0) * real(i-1) / real(num_ell-1), i = 1, num_ell)]
    
    ! Define the range of theta values
    theta = 10.0 ** [(log10(theta_min) + (log10(theta_max) - log10(theta_min)) * real(i-1) / real(num_theta-1), i = 1, num_theta)]
    
    ! Compute xi_plus(theta) for each theta value
    do i = 1, num_theta
        xi_plus(i) = 0.0
        do j = 1, num_ell
            xi_plus(i) = xi_plus(i) + quadgk(integrand(theta(i), ell(j)), 10.0, 100000.0)
        end do
        xi_plus(i) = xi_plus(i) / (2 * pi)
    end do
    
    ! Output
    open(unit=10, file="xi_plus.txt", status="replace")
    do i = 1, num_theta
        write(10, '(2E20.10)') theta(i), xi_plus(i)
    end do
    close(10)
    
contains

    real function integrand(theta, ell)
        real, intent(in) :: theta, ell
        integrand = P_kappa(j) * besselj0(theta * ell) * ell
    end function integrand
    
end program TwoPointCorrelation

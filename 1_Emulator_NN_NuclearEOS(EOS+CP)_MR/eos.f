       SUBROUTINE eos(aL_in,aKsym_in,rho1_in,rho2_in,
     & rho3_in,Gamma1,Gamma2,Gamma3,icount,rnb,pres,ener,cs2,iflag)


      implicit real*8(a-h,o-z)
      parameter(nmax=5000)
      real*8 rnb(0:nmax),ener(0:nmax),pres(0:nmax),cs2(0:nmax)
      common/consth1/rnb0,cgs1,cgs2,baryon_mass,rho_rnb0
      common/param/aK0,esym0,aL,aKsym
      common/sly/aKL_1,GammaL_1

cf2py intent(out) icount,rnb,ener,pres,cs2,iflag

      aL=aL_in
      aKsym=aKsym_in
      esym0=31.7d0
      aK0 = 240.0

      !******************************************************
      !           Defining some useful constants            * 
      !******************************************************
      cgs1=        1.7827D+12                  ! MeV/fm3 to gms/cm3
      cgs2=        1.6022D+33                  ! MeV/fm3 to dyne/cm2
      rnb0=        0.16D0                      ! in fm^-3
      baryon_mass= 931.494D0                   ! in Mev
      rho_rnb0=    rnb0*baryon_mass*cgs1       ! in gm/cm^3

      !******************************************************
      !  fixed (cold) crust from Read et al (2009)          * 
      !******************************************************

      rhoL_1 = 2.62789d+12
      rhoL_2 = 3.78358d+11
      rhoL_3 = 2.44034d+7
      rhoL_4 = 0.0

      GammaL_1 = 1.35692
      GammaL_2 = 0.62223
      GammaL_3 = 1.28733
      GammaL_4 = 1.58425

      aKL_1 = 3.99874d-8
      aKL_2 = 5.32697d+1
      aKL_3 = 1.06186d-6
      aKL_4 = 6.80110d-9

cccccc  High density EoS   cccccccc

      rho1 = rho1_in*rho_rnb0
      rho2 = rho2_in*rho_rnb0
      rho3 = rho3_in*rho_rnb0

      epsL_4 = 0.0
      alphaL_4 = 0.0
      
      epsL_3 = (1+alphaL_4)*rhoL_3 +
     &         aKL_4/(GammaL_4 - 1)*rhoL_3**GammaL_4
      alphaL_3 = epsL_3/rhoL_3 - 1 -
     &           aKL_3/(GammaL_3 - 1)*rhoL_3**(GammaL_3 -1)
     
      epsL_2 = (1+alphaL_3)*rhoL_2 +
     &         aKL_3/(GammaL_3 - 1)*rhoL_2**GammaL_3
      alphaL_2 = epsL_2/rhoL_2 - 1 -
     &           aKL_2/(GammaL_2 - 1)*rhoL_2**(GammaL_2 -1)
     
      epsL_1 = (1+alphaL_2)*rhoL_1 +
     &         aKL_2/(GammaL_2 - 1)*rhoL_1**GammaL_2
      alphaL_1 = epsL_1/rhoL_1 - 1 -
     &           aKL_1/(GammaL_1 - 1)*rhoL_1**(GammaL_1 -1)


      prs_rho1 = empirical_prs(rho1)

      aK1 = prs_rho1*rho1**(-Gamma1)
      aK2 = aK1 * rho2**(Gamma1-Gamma2)
      aK3 = aK2 * rho3**(Gamma2-Gamma3)


      eps1 = empirical_eng(rho1)

      alpha1 = eps1/rho1 - 1 - aK1/(Gamma1 - 1)*rho1**(Gamma1 -1)

      eps2 = (1+alpha1)*rho2 + aK1/(Gamma1 - 1)*rho2**Gamma1
      alpha2 = eps2/rho2 - 1 - aK2/(Gamma2 - 1)*rho2**(Gamma2 -1)

      eps3 = (1+alpha2)*rho3 + aK2/(Gamma2 - 1)*rho3**Gamma2
      alpha3 = eps3/rho3 - 1 - aK3/(Gamma3 - 1)*rho3**(Gamma3 -1)



      step1=0.1d0
      step2=0.01d0
      rhomin = 1e-12*rho_rnb0  ! in gm/cm^3
      rhomax = 8.3*rho_rnb0 ! in gm/cm^3
      rho = rhomin
      icount = 0

ccc---Determination of crust-core junction density (rho0)-----------

      e = 1d-12
      x1= rhoL_1
      x2= rho1
      CALL Bisection(f,e,m,x0,x1,x2,iflag)
      rho0 = x0
c-----------------

15	  continue

      if(rho.lt.rhoL_3)then
	  prs = aKL_4*rho**GammaL_4 
	  eng = (1.0+alphaL_4)*rho + aKL_4/(GammaL_4-1.0)*rho**GammaL_4
	  sos= GammaL_4*prs/(prs+eng)	  
      else if(rho.ge.rhoL_3.and.rho.lt.rhoL_2)then
	  prs = aKL_3*rho**GammaL_3 
	  eng = (1.0+alphaL_3)*rho + aKL_3/(GammaL_3-1.0)*rho**GammaL_3
	  sos= GammaL_3*prs/(prs+eng)
      else if(rho.ge.rhoL_2.and.rho.lt.rhoL_1)then
	  prs = aKL_2*rho**GammaL_2
	  eng = (1.0+alphaL_2)*rho + aKL_2/(GammaL_2-1.0)*rho**GammaL_2
	  sos= GammaL_2*prs/(prs+eng)
      else if(rho.ge.rhoL_1.and.rho.lt.rho0)then
	  prs = aKL_1*rho**GammaL_1 
	  eng = (1.0+alphaL_1)*rho + aKL_1/(GammaL_1-1.0)*rho**GammaL_1
	  sos= GammaL_1*prs/(prs+eng)
      else if(rho.ge.rho0.and.rho.lt.rho1)then
	  prs =empirical_prs(rho)
	  eng = empirical_eng(rho)
	  sos= empirical_cs2(rho)
      else if(rho.ge.rho1.and.rho.lt.rho2)then
	  prs = aK1*rho**Gamma1
	  eng = (1.0+alpha1)*rho + aK1/(Gamma1-1.0)*rho**Gamma1
	  sos= Gamma1*prs/(prs+eng)
      else if(rho.ge.rho2.and.rho.lt.rho3)then
	  prs = aK2*rho**Gamma2
	  eng = (1.0+alpha2)*rho + aK2/(Gamma2-1.0)*rho**Gamma2
	  sos= Gamma2*prs/(prs+eng)
      else if(rho.ge.rho3)then
	  prs = aK3*rho**Gamma3
	  eng = (1.0+alpha3)*rho + aK3/(Gamma3-1.0)*rho**Gamma3
	  sos= Gamma3*prs/(prs+eng)
      endif



      rnb(icount)=  rho/rho_rnb0	! rho/rho_0
      pres(icount)= prs/cgs1 	        !in MeV/fm^3
      ener(icount)= eng/cgs1 	        !in MeV/fm^3
      cs2(icount)=  sos

      !******************************************************
      !     Pressure decreases; mechanical instability      *
      !******************************************************
      if (pres(icount).lt.pres(icount-1)) then
        goto 100	
      endif
      
      !******************************************************
      !           Causality violation condition             *
      !******************************************************
      if (cs2(icount).gt.1.0) then
        goto 100		
      endif



      if(rho.lt.rho_rnb0)then
        rho=rho*exp(step1)
      else if(rho.ge.rho_rnb0)then
        rho=rho*exp(step2)
      endif

      icount = icount + 1

      if(rho.lt.rhomax) go to 15


100   continue



      contains

      !**************************************************
      ! pressure bisection routine                      !
      ! to determine crust-core transition density      !
      !*************************************************!

      Subroutine Bisection(f,e,m,x,x0,x1,flag)
      integer m,flag
      real(kind=8) :: f,e,x,x0,x1,y0,yy,abs_err
      m=0

      if (f(x0) == 0.d0) then
      flag = 0
      x = x0
      return
      else if (f(x1) == 0.d0) then
      flag = 0
      x = x1
      return
      else if (f(x0)*f(x1) > 0.d0) then
      flag = 1
      return
      else if (f(x0)*f(x1) < 0.d0) then
      flag = 0
      abs_err = dabs((x1-x0)/dabs(x0))
      do while (abs_err > e)
      y0=f(x0)
      x=(x0+x1)/2.d0
      yy=f(x)
      m=m+1
      if (dabs(yy*y0) == 0.d0) return
      if ((yy*y0)<0.d0) x1=x
      if ((yy*y0)>0.d0) x0=x
      abs_err = dabs((x1-x0)/dabs(x0))
      enddo
      return
      endif
      end Subroutine Bisection

      !**************************************************
      ! pressure bisection function which is needed     !
      ! to determine crust-core transition density      !
      !*************************************************!

      double precision function f(rho)
      implicit real*8(a-h,o-z)
      common/consth1/rnb0,cgs1,cgs2,baryon_mass,rho_rnb0
      common/sly/aKL_1,GammaL_1
      prs = aKL_1*rho**GammaL_1
      f = prs - empirical_prs(rho)

      end function f
      
      !**************************************************
      ! This function computes the pressure for         !
      ! nuclear empirical EOS in gm/cm^3 unit           !
      !*************************************************!

      double precision function empirical_prs(rho)
      implicit real*8(a-h,o-z)
      common/consth1/rnb0,cgs1,cgs2,baryon_mass,rho_rnb0
      common/param/aK0,esym0,aL,aKsym

      rr = (rho/rho_rnb0 - 1.)/3.
      del = 1.-2.*yp(rho)
      psnm= aK0*rr*(rho/rho_rnb0)**2*rnb0/3.0
      psym = ((aL + aKsym*rr)*del*del)*(rho/rho_rnb0)**2*rnb0/3.0
      empirical_prs = (psnm + psym)*cgs1  ! gm/cm^3

      end function empirical_prs
      
      
      !**************************************************
      ! This function computes the energy density for   !
      ! nuclear empirical EOS in gm/cm^3 unit           !
      !*************************************************!

      double precision function empirical_eng(rho)
      implicit real*8(a-h,o-z)
      common/consth1/rnb0,cgs1,cgs2,baryon_mass,rho_rnb0
      common/param/aK0,esym0,aL,aKsym


      rr = (rho/rho_rnb0 - 1.)/3.
      del = 1.-2.*yp(rho)
      e00 = -15.9d0
      e0 = e00 + aK0*rr*rr/2.0
      esym = esym0 + aL*rr + aKsym*rr*rr/2.0
      epp=e0+esym*del*del
      empirical_eng = (rho*rnb0/rho_rnb0)*(epp + baryon_mass)*cgs1  ! gm/cm^3

      end function empirical_eng
      
      !**************************************************
      ! This function computes the speed of sound for   !
      ! nuclear empirical EOS                           !
      !*************************************************!
      
      double precision function empirical_cs2(rho)
      implicit real*8(a-h,o-z)
      common/consth1/rnb0,cgs1,cgs2,baryon_mass,rho_rnb0,aLambda_emp
      common/param/aK0,esym0,aL,aKsym


      rr = (rho/rho_rnb0 - 1.)/3.
      del = 1.-2.*yp(rho)
      e00 = -15.9d0
      e0 = e00 + aK0*rr*rr/2.0
      esym = esym0 + aL*rr + aKsym*rr*rr/2.0
      epp=e0+esym*del*del
      eng_tot = epp + baryon_mass
      
      empirical_cs2= (2.*rho*(3.*rho-2.*rho_rnb0)
     &               *(aK0+aKsym*del*del)/(9.*rho_rnb0**2)
     &               + 2.*aL*rho*del*del/(3.*rho_rnb0))/
     &               (2.*rho*(rho-rho_rnb0)
     &               *(aK0+aKsym*del*del)/(9.*rho_rnb0**2)
     &               + aL*rho*del*del/(3.*rho_rnb0) + eng_tot
     &               - aLambda_emp/rho)
      end function empirical_cs2

      !**************************************************
      ! A function which computes electron fraction for !
      ! cold neutron star matter implementing beta      !
      ! equilibrium                                     !
      !*************************************************!
      
      double precision function yp(rho)
      implicit real*8(a-h,o-z)
      common/consth1/rnb0,cgs1,cgs2,baryon_mass,rho_rnb0
      common/param/aK0,esym0,aL,aKsym
      pi=4.d0*atan(1.d0)
      pi2=pi*pi
      c = 2.9979d+8  ! in m/s
      h = 6.626d-34 ! in m^2 kg / s
      rMeV = 1.6d-13   ! kg m^2 s^−2
      fm = 1d-15 ! fm to m

      rr = (rho/rho_rnb0 - 1.)/3.
      esym = esym0 + aL*rr + aKsym*rr*rr/2.0
      if(esym.lt.0.0d0)then
      yp = 0.0
	    else
      rn = rho*rnb0/rho_rnb0
      x = (2.*pi*esym*rMeV/(h*c))
      epsilon = x**2*(24.*rn*fm**(-3.)*(1.+
     &    sqrt(1.+pi2*rn*fm**(-3.)/(288.*x**3))))**(1./3.)

      yp = 0.5 + ((2.*pi2)**(1./3.)/32.)*(rn*fm**(-3.)/epsilon)*
     &    ((2.*pi2)**(1./3.)-epsilon**2/(x**3*rn*fm**(-3.)))

      if (yp.lt.0.0) then
      yp = 0.0
      endif
      endif

      end function yp


      END SUBROUTINE eos

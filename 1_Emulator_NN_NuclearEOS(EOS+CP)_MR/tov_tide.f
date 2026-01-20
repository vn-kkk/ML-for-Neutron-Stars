C 	Tidal deformability calculation
      SUBROUTINE tov_tide(e, p, pc, m, r, lambda, N)


      implicit none
      real*8 e(N),p(N), pc, m, r, lambda, dr
      real*8 k1,k2,k3,k4,l1,l2,l3,l4,m1,m2,m3,m4,h
      real*8 sr,se,sm,c,y,ak2,td,td1,td2
      integer N,i,k,imax

c      common/ans/  pfunc, emfunc
      real*8 pi
      common/const/ pi

      integer, parameter :: l=100000, NC1 =1000

      real*8 p_prf(0:l),e_prf(0:l),em_prf(0:l),rad(0:l),y_prf(0:l)
      real*8 p_lim, pc_min, pc_min0, pc_max, intp, newpc_max, xmsun
      real*8 pfunc, emfunc, yfunc



Cf2py intent(out) m, r, lambda


      pi = dacos(-1.0d0)
      xmsun = 1.4766d0      !! mass of sun (in km)
      dr = 0.0001d0

      p_lim = 5.0d-18 !pres(1)  ! end point of TOV integration, lowest pressure limit
      pc_min = 1.3234d-9

      pc_max = p(N)



C-------------------------------------

       intp = exp((log(pc_max)-log(pc_min))/NC1)

c       do k= 1,NC1

c  start to calculate the mass and radius of a star at a given ener and prs
C-------------RK4 Implementation:

C	First Step..............

        p_prf(0)= pc
        e_prf(0)= eng(pc,e,p,N)

        y_prf(0)=2.d0		! central value of y: y(0)=2.0 for l=2

	em_prf(0)= 0.0
	rad(0)= 1.0d-6

c	k1=0.d0
c	l1=0.d0
c	m1=0.d0
	h=dr

	call structure(rad(0), p_prf(0), em_prf(0), y_prf(0), e, p, N,
     &			eng, vs, pfunc, emfunc, yfunc)

	k1=h*pfunc
	l1=h*emfunc
	m1=h*yfunc

	call structure(rad(0)+h/2.d0, p_prf(0)+k1/2.d0,
     &			em_prf(0)+l1/2.d0, y_prf(0)+m1/2.d0, e, p, N,
     &			 eng, vs, pfunc, emfunc, yfunc)

	k2=h*pfunc
	l2=h*emfunc
	m2=h*yfunc

	call structure(rad(0)+h/2.d0, p_prf(0)+k2/2.d0,
     &			em_prf(0)+l2/2.d0, y_prf(0)+m2/2.d0, e, p, N,
     &			 eng, vs, pfunc, emfunc, yfunc)

	k3=h*pfunc
	l3=h*emfunc
	m3=h*yfunc

	call structure(rad(0)+h, p_prf(0)+k3,
     &		        em_prf(0)+l3, y_prf(0)+m3, e, p, N,
     &			 eng, vs, pfunc, emfunc, yfunc)

	k4=h*pfunc
	l4=h*emfunc
	m4=h*yfunc

	p_prf(1)=p_prf(0)+(k1+2.d0*k2+2.d0*k3+k4)/6.d0
	em_prf(1)=em_prf(0)+(l1+2.d0*l2+2.d0*l3+l4)/6.d0
	y_prf(1)=y_prf(0)+(m1+2.d0*m2+2.d0*m3+m4)/6.d0

	rad(1)=rad(0)+h
	e_prf(1)=eng(p_prf(1), e, p, N)


C	Next Steps..............

	do i=1,l-1


	imax=i+1

	call structure(rad(i), p_prf(i), em_prf(i), y_prf(i), e, p, N,
     &			eng, vs, pfunc, emfunc, yfunc)

	h=0.01d0/(emfunc/em_prf(i) - pfunc/p_prf(i))

	k1=h*pfunc
	l1=h*emfunc
	m1=h*yfunc

	call structure(rad(i)+h/2.d0, p_prf(i)+k1/2.d0,
     &			em_prf(i)+l1/2.d0, y_prf(i)+m1/2.d0, e, p, N,
     &			 eng, vs, pfunc, emfunc, yfunc)

	k2=h*pfunc
	l2=h*emfunc
	m2=h*yfunc

	call structure(rad(i)+h/2.d0, p_prf(i)+k2/2.d0,
     &			em_prf(i)+l2/2.d0, y_prf(i)+m2/2.d0, e, p, N,
     &			 eng,vs, pfunc, emfunc, yfunc)

	k3=h*pfunc
	l3=h*emfunc
	m3=h*yfunc

	call structure(rad(i)+h, p_prf(i)+k3,
     &                   em_prf(i)+l3, y_prf(i)+m3, e, p, N,
     &			 eng, vs, pfunc, emfunc, yfunc)

	k4=h*pfunc
	l4=h*emfunc
	m4=h*yfunc

	p_prf(i+1)=p_prf(i)+(k1+2.d0*k2+2.d0*k3+k4)/6.d0
	em_prf(i+1)=em_prf(i)+(l1+2.d0*l2+2.d0*l3+l4)/6.d0
	y_prf(i+1)=y_prf(i)+(m1+2.d0*m2+2.d0*m3+m4)/6.d0

	rad(i+1)=rad(i)+h
	e_prf(i+1)=eng(p_prf(i+1), e, p, N)

	if(p_prf(i+1).le.p_lim)goto 32



	enddo

 32	continue

c        pc(k) = pc_min*(intp**(k-1))
        m = em_prf(imax)/xmsun
        r = rad(imax)



	sr=rad(imax)
        se=e_prf(imax)
        sm=em_prf(imax)
	c=sm/sr
C! The following line is added to handle density discontinuity on star's surface (eqn. 15 of PRD 81, 123016 (2010))
	y=y_prf(imax) - 4.d0*pi*(sr**3.d0)*se/sm

	ak2 = 8.d0*(c**5.d0)*((1.d0-2.d0*c)**2.d0)*(2.d0+2.d0*c*(y-1.d0)
     &      -y)/5.d0/(2.d0*c*(6.d0-3.d0*y+3.d0*c*(5.d0*y-8.d0))
     &	    +4.d0*(c**3.d0)*(13.d0-11.d0*y+c*(3.d0*y-2.d0)
     &      +2.d0*(c**2.d0)*(1.d0+y))+3.d0*((1.d0-2.d0*c)**2.d0)*(2.d0-y
     &	    +2.d0*c*(y-1.d0))*dlog(1.d0-2.d0*c))

	td= (2.d0/3.d0)*ak2*sr**5.d0
	td1= td*1.0e-4
	td2= td/(sm**5.d0) ! dimensionless Lambda

	lambda=td2

c        enddo

c 51   continue




      contains

C--------- Interpolation functions

C	energy density as a function of pressure
      	real*8 function eng(ap,ener,pres,N)

      	real*8 ener(N), pres(N), ap, eng1
        real*8 xp1, xe1, xp2, xe2
        integer N, i1

	do i1=1,N
	xp1 = pres(i1)
	xe1 = ener(i1)
	xp2 = pres(i1+1)
	xe2 = ener(i1+1)
	if(ap.ge.xp1.and.ap.le.xp2)then
	eng1= xe1*dexp(dlog(ap/xp1)*dlog(xe2/xe1)/dlog(xp2/xp1))
	goto 11
	endif ! logarthmic interpolation
     	enddo
  11	eng=eng1

	end


C	sound speed squared as a function of pressure
	real*8 function vs(zp,ener,pres,N)

      	real*8 ener(N), pres(N), zp, vs1
        real*8 zp1, ze1, zp2, ze2
        integer N, i1

 	do i1=1,N
	zp1 = pres(i1)
	ze1 = ener(i1)
	zp2 = pres(i1+1)
	ze2 = ener(i1+1)
	if(zp.ge.zp1.and.zp.le.zp2)then
	vs1= (zp2-zp1)/(ze2-ze1) ! (zp1*dlog(zp2/zp1))/(ze1*dlog(ze2/ze1))
	goto 12
	endif
     	enddo
  12	vs=vs1

	end

C--------- Calculation of structure

	subroutine structure(xr,xp,xm,xy,ener,pres,N,
     &		f1,f2,pfc,emfc,yfc)

	integer N
        real*8 ener(N), pres(N)
        real*8 xr,xp,xm,xy
        real*8 xe,cs,vol,ycoef1,ycoef2,yterm1,yterm2,yterm3
        real*8 f1,f2
        real*8 pfc, emfc, yfc
        intent(out) pfc, emfc, yfc

c      	common/ans/  pfunc, emfunc
        real*8 pi
      	common/const/ pi

      	vol = 4.*pi*xr*xr*xr

	xe = f1(xp, ener, pres, N)

	cs = f2(xp, ener, pres, N)

      	pfc= -(xe+xp)*(xm + vol*xp)/(xr*(xr-2.*xm))

      	emfc= 4.*pi*xr*xr*xe

	ycoef1= -1./xr

	ycoef2= -(xr + vol*(xp-xe))/(xr*(xr-2.*xm))

	yterm1= (4.*(xm + vol*xp)**2.)/(xr*(xr-2.*xm)**2.)

	yterm2= 6./(xr-2.*xm)

        yterm3= -4.*pi*xr*xr*(5.*xe + 9.*xp
     &		+ (xe+xp)/cs)/(xr-2.*xm)

	yfc= ycoef1*xy*xy + ycoef2*xy + yterm1 + yterm2 + yterm3


      	end

       END

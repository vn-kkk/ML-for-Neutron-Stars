C FILE: TOV_sequence.F
      SUBROUTINE TOV(e, p, pc, m, r, N)


      real*8 e(N),p(N), pc, m, r, dr
      real*8 k1,k2,k3,k4,l1,l2,l3,l4,h
      integer N

c      common/ans/  pfunc, emfunc
      common/const/ pi

      integer, parameter :: l=100000, NC1 =1000


      real*8 p_prf(0:l),e_prf(0:l),em_prf(0:l),rad(0:l)
      real*8 p_lim, pc_min, pc_max, intp
      real*8 pfunc, emfunc

Cf2py intent(out) m, r

      pi = dacos(-1.0d0)
      xmsun = 1.4766d0      !! mass of sun (in km)
      dr = 0.0001d0

      p_lim = 5.0d-18 !pres(1)  ! end point of TOV integration, lowest pressure limit
      pc_min = 1.3234d-9
      pc_max = p(N)

       intp = exp((log(pc_max)-log(pc_min))/NC1)

c       do k= 1,NC1

c  start to calculate the mass and radius of a star at a given ener and prs
C-------------RK4 Implementation:

C	First Step..............

        p_prf(0)= pc
        e_prf(0)= eng(pc,e,p,N)

	em_prf(0)= 0.0
	rad(0)= 1.0d-6

c	k1=0.d0
c	l1=0.d0
	h=dr

	call structure(rad(0),p_prf(0),em_prf(0), e, p, N,
     &			eng,pfunc,emfunc)


	k1=h*pfunc
	l1=h*emfunc

	call structure(rad(0)+h/2.d0, p_prf(0)+k1/2.d0,
     &			em_prf(0)+l1/2.d0, e, p, N, eng,pfunc,emfunc)

	k2=h*pfunc
	l2=h*emfunc

	call structure(rad(0)+h/2.d0, p_prf(0)+k2/2.d0,
     &			em_prf(0)+l2/2.d0, e, p, N, eng,pfunc,emfunc)

	k3=h*pfunc
	l3=h*emfunc

	call structure(rad(0)+h, p_prf(0)+k3,
     &		        em_prf(0)+l3, e, p, N, eng,pfunc,emfunc)

	k4=h*pfunc
	l4=h*emfunc

	p_prf(1)=p_prf(0)+(k1+2.d0*k2+2.d0*k3+k4)/6.d0
	em_prf(1)=em_prf(0)+(l1+2.d0*l2+2.d0*l3+l4)/6.d0
	rad(1)=rad(0)+h
	e_prf(1)=eng(p_prf(1), e, p, N)


C	Next Steps..............

	do i=1,l-1


	imax=i+1

	call structure(rad(i),p_prf(i),em_prf(i), e, p, N,
     &			eng,pfunc,emfunc)

	h=0.1d0/(emfunc/em_prf(i) - pfunc/p_prf(i))

	k1=h*pfunc
	l1=h*emfunc

	call structure(rad(i)+h/2.d0, p_prf(i)+k1/2.d0,
     &			em_prf(i)+l1/2.d0, e, p, N, eng,pfunc,emfunc)

	k2=h*pfunc
	l2=h*emfunc

	call structure(rad(i)+h/2.d0, p_prf(i)+k2/2.d0,
     &			em_prf(i)+l2/2.d0, e, p, N, eng,pfunc,emfunc)

	k3=h*pfunc
	l3=h*emfunc

	call structure(rad(i)+h, p_prf(i)+k3,
     &                   em_prf(i)+l3, e, p, N, eng,pfunc,emfunc)

	k4=h*pfunc
	l4=h*emfunc

	p_prf(i+1)=p_prf(i)+(k1+2.d0*k2+2.d0*k3+k4)/6.d0
	em_prf(i+1)=em_prf(i)+(l1+2.d0*l2+2.d0*l3+l4)/6.d0
	rad(i+1)=rad(i)+h
	e_prf(i+1)=eng(p_prf(i+1), e, p, N)

	if(p_prf(i+1).le.p_lim)goto 32



	enddo

 32	continue

c        pc = pc_min*(intp**(k-1))
        m = em_prf(imax)/xmsun
        r = rad(imax)

c        enddo



      contains

C--------- Interpolation functions

      	real*8 function eng(ap,ener,pres,N)	! energy density as a function of pressure

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

C--------- Calculation of structure

	subroutine structure(xr,xp,xm, ener, pres, N, f1, pfc, emfc)

        real*8 ener(N), pres(N)
        real*8 xr, xp, xm, vol
        real*8 f1
        real*8 pfc, emfc
        intent(out) pfc, emfc

c      	common/ans/  pfunc, emfunc
      	common/const/ pi

      	vol = 4.*pi*xr*xr*xr

      	pfc= -(f1(xp, ener, pres, N)+xp)*(xm + vol*xp)/(xr*(xr-2.*xm))

      	emfc= 4.*pi*xr*xr*f1(xp, ener, pres, N)

      	end

       END
C END FILE TOV_Sequence.F

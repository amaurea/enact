module nmat_core
	implicit none
! This module implements the core of the inverse noise matrix
! of the map maker, in order to make it openmp-able. In the
! previous version of the noise matrix, evaluation was not a
! real bottleneck, but it took about 1/3 of the time, I think.
! It would be nice to get that down to negligible amounts.
! As with pmat_core, this module handles one TOD at a time.
! Loop over tods in python, calling this for each.

! Because of the possibility of a variable number of noise
! basis vectors per bin, these are stored in a flattened format:
! (ndet,nvec)+(nbin), and similarly for the mode noise.

! For each frequency we will perform
!  N"d = (Nu+VEV')"d
! By the woodbury identity, this is
!  N"d = (Nu"-Nu"V(E"+V'Nu"V)"V'Nu")d
! d is pretty big, so we want as few operations with that as possible.
! On the other hand, we don't want to expand the full ndet*ndet matrices
!  Q' = sqrt((E"+V'Nu"V)")V'Nu"
! This results in only 3 multiplications rather than 5 as one
! would otherwise get. We therefore require Q rather than V and
! E as arguments. Q us (ndet,nvec), just like  is.

contains

	subroutine nmat(ftod, bins, iNu, Q, vbins)
		implicit none
		! Arguments
		complex(4), intent(inout) :: ftod(:,:)
		integer(4), intent(in)    :: bins(:,:), vbins(:,:)
		real(4),    intent(in)    :: iNu(:,:), Q(:,:)
		! Work
		complex(4), allocatable   :: cQ(:,:), Qd(:,:), orig(:,:), iNud(:,:)
		integer(4)                :: bi, nbin, nfreq, ndet, b1, b2, di, fi, nv, vi, nf,v1,v2
		integer(4)                :: info
		nfreq = size(ftod,1)
		ndet  = size(ftod,2)
		nbin  = size(bins,2)

		! Q is (ndet,nvec) for slicing efficiency reasons
		allocate(cQ(size(Q,1),size(Q,2)))
		cQ = Q

		!$omp parallel do private(bi,b1,b2,v1,v2,nf,nv,Qd,orig,iNud,di) schedule(dynamic)
		do bi = nbin, 1, -1
			b1 = bins(1,bi)+1;   b2 = bins(2,bi)
			b1 = min(b1, nfreq); b2 = min(b2,nfreq)
			v1 = vbins(1,bi)+1;  v2 = vbins(2,bi)
			nf = b2-b1+1; nv = v2-v1+1
			if(nf < 1) continue ! Skip empty bins
			allocate(Qd(nv,nf)) ! Q'd'
			allocate(orig(nf,ndet), iNud(nf,ndet))
			Qd = 0; iNud = 0
			orig = ftod(b1:b2,:)
			do di = 1, ndet
				iNud(:,di) = orig(:,di)*iNu(di,bi)
			end do
			if(nv > 0) then
				! Q'd' = matmul(transpose(Q)(nvec,ndet),transpose(ftod)(ndet,nf))
				! QQ'd = matmul(Q(ndet,nvec),Qd(nvec,nf))
				! => (QQ'd)' = matmul(transpose(Qd)(nf,nvec),transpose(Q)(nvec,det))
				call cgemm('T', 'T', nv, nf, ndet, (1.0,0.0), cQ(:,v1:v2), ndet, orig, nf, (0.0,0.0), Qd, nv)
				call cgemm('T', 'T', nf, ndet, nv,(-1.0,0.0), Qd, nv, cQ(:,v1:v2), ndet,   (1.0,0.0), iNud, nf)
			end if
			do di = 1, ndet
				ftod(b1:b2,di) = iNud(:,di)
			end do
			deallocate(Qd, orig, iNud)
		end do
	end subroutine

	subroutine nwhite(tod, bins, iNu)
		implicit none
		! Arguments
		real(4),    intent(inout) :: tod(:,:)
		integer(4), intent(in)    :: bins(:,:)
		real(4),    intent(in)    :: iNu(:,:)
		! Work
		integer(4) :: di
		!$omp parallel do
		do di = 1, size(tod,2)
			tod(:,di) = tod(:,di) * sum(iNu(di,:)*(bins(2,:)-bins(1,:)))/sum(bins(2,:)-bins(1,:))
		end do
	end subroutine
end module

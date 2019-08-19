from libc.math cimport sqrt
ctypedef unsigned long long uint64_t

def find_primes(uint64_t n):
	assert n>=2, 'input should be >=2'

	primes=[2]
	cdef:
		int is_prime=1
		uint64_t candidate=3 #start from odd number
		uint64_t num
		uint64_t cmax= <uint64_t>sqrt(candidate)+1

	while(candidate<n):
		print(candidate,cmax)
		num=2
		while is_prime & num<=cmax:
			if candidate%num==0:
				print('{}%{}=0'.format(candidate,num))
				is_prime=0
			num+=1

		if is_prime:
			primes.append(candidate)
			candidate+=2 #check every odd number
			cmax= <uint64_t> sqrt(candidate)+1
			is_prime=1

	return primes
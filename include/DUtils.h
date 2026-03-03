#ifndef __DUTILS_RANDOM_COMPAT__
#define __DUTILS_RANDOM_COMPAT__

#include <cstdlib>
#include <vector>
using namespace std;
namespace DUtils {

class Random
{
public:
    static void SeedRand()
    {
        srand((unsigned int)time(nullptr));
    }

    static void SeedRandOnce()
    {
        if(!m_already_seeded)
        {
            SeedRand();
            m_already_seeded = true;
        }
    }

    static void SeedRand(int seed)
    {
        srand(seed);
    }

    static void SeedRandOnce(int seed)
    {
        if(!m_already_seeded)
        {
            SeedRand(seed);
            m_already_seeded = true;
        }
    }

    static int RandomInt(int min, int max)
    {
        int d = max - min + 1;
        return int(((double)rand()/((double)RAND_MAX + 1.0)) * d) + min;
    }

    template <class T>
    static T RandomValue()
    {
        return (T)rand()/(T)RAND_MAX;
    }

    template <class T>
    static T RandomValue(T min, T max)
    {
        return RandomValue<T>() * (max - min) + min;
    }

private:
    static inline bool m_already_seeded = false;
};

}

#endif

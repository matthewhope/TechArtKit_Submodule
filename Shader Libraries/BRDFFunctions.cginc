/*
 * The Dirtchamber - Tobias Alexander Franke 2013
 * For copyright and license see LICENSE
 * http://www.tobias-franke.eu
 */

#define M_PI 3.141592

#ifndef BRDF_HLSL
#define BRDF_HLSL

#define F0_WATER        0.02,0.02,0.02
#define F0_PLASTIC      0.03,0.03,0.03
#define F0_PLASTIC_HIGH 0.05,0.05,0.05
#define F0_GLASS        0.08,0.08,0.08
#define F0_DIAMOND      0.17,0.17,0.17
#define F0_IRON         0.56,0.57,0.58
#define F0_COPPER       0.95,0.64,0.54
#define F0_GOLD         1.00,0.71,0.29
#define F0_ALUMINIUM    0.91,0.92,0.92
#define F0_SILVER       0.95,0.93,0.88

float sqr(float x)
{
    return x * x;
}

// Schlick's approximation of the fresnel term
float F_schlick(float f0, float LoH)
{
    // only have specular if f0 isn't 0
    //float enable = float(dot(f0, 1.0f) > 0.0f);
    return (f0 + (1.0f - f0) * pow(1.0f - LoH, 5.0f));
}

// Optimizied Schlick
// http://seblagarde.wordpress.com/2011/08/17/hello-world/
float SphericalGaussianApprox(float CosX, float ModifiedSpecularPower)
{
    return exp2(ModifiedSpecularPower * CosX - ModifiedSpecularPower);
}

#define OneOnLN2_x6 8.656170 // == 1/ln(2) * 6   (6 is SpecularPower of 5 + 1)

float3 F_schlick_opt(float3 SpecularColor, float3 E, float3 H)
{
    // In this case SphericalGaussianApprox(1.0f - saturate(dot(E, H)), OneOnLN2_x6) is equal to exp2(-OneOnLN2_x6 * x)
    return SpecularColor + (1.0f - SpecularColor) * exp2(-OneOnLN2_x6 * saturate(dot(E, H)));
}

// Microfacet Models for Refraction through Rough Surfaces
// Walter et al.
// http://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.html
// aka Towbridge-Reitz
float D_ggx(float alpha, float NoH)
{
    float a2 = alpha*alpha;
    float cos2 = NoH*NoH;

    return (1.0f/M_PI) * sqr(alpha/(cos2 * (a2 - 1) + 1));

    /*
    // version from the paper, eq 33
    float CosSquared = NoH*NoH;
    float TanSquared = (1.0f - CosSquared)/CosSquared;
    return (1.0f/M_PI) * sqr(alpha/(CosSquared * (alpha*alpha + TanSquared)));
    */
}

// Smith GGX with denominator
// http://graphicrants.blogspot.co.uk/2013/08/specular-brdf-reference.html
float G_smith_ggx(float a, float NoV, float NoL)
{
    float a2 = a*a;
    float G_V = NoV + sqrt((NoV - NoV * a2) * NoV + a2);
    float G_L = NoL + sqrt((NoL - NoL * a2) * NoL + a2);
    return rcp(G_V * G_L);
}

// Schlick GGX
// http://graphicrants.blogspot.co.uk/2013/08/specular-brdf-reference.html
float G_UE4(float alpha, float NoV)
{
    float k = alpha/2;
    return NoV/(NoV * (1 - k) + k);
}

float G_implicit(float NoV, float NoL)
{
    return NoL * NoV;
}

// Beckmann distribution
float D_beckmann(float m, float t)
{
    float M = m*m;
    float T = t*t;
    return exp((T-1)/(M*T)) / (M*T*T);
}

// Helper to convert roughness to Phong specular power
float alpha_to_spec_pow(float a)
{
    return 2.0f / (a * a) - 2.0f;
}

// Helper to convert Phong specular power to alpha
float spec_pow_to_alpha(float s)
{
    return sqrt(2.0f / (s + 2.0f));
}

// Blinn Phong with conversion functions for roughness
float D_blinn_phong(float n, float NoH)
{
    float alpha = spec_pow_to_alpha(n);

    return (1.0f / (M_PI*alpha*alpha)) * pow(NoH, n);
}

// Cook-Torrance specular BRDF + diffuse
float3 brdf(float3 L, float3 V, float3 N, float3 cdiff, float3 cspec, float roughness)
{
    float alpha = roughness*roughness;

    float3 H = normalize(L+V);

    float NoL = dot(N, L);
    float NoV = dot(N, V);
    float NoH = dot(N, H);
    float LoH = dot(L, H);

    // refractive index
    float n = 1.5;
    float f0 = pow((1 - n)/(1 + n), 2);

    // the fresnel term
    float F = F_schlick(f0, LoH);

    // the geometry term
    float G = G_UE4(alpha, NoV);

    // the NDF term
    float D = D_ggx(alpha, NoH);

    // specular term
    float3 Rs = cspec/M_PI *
                (F * G * D)/
                (4 * NoL * NoV);

    // diffuse fresnel, can be cheaper as 1-f0
    float Fd = F_schlick(f0, NoL);

    float3 Rd = cdiff/M_PI * (1.0f - Fd);

    return (Rd + Rs);
}

#endif


// Following BRDF methods are based upon research Frostbite EA
//[Lagrade et al. 2014, "Moving Frostbite to Physically Based Rendering"]

//Schlick Fresnel
//specular  = the rgb specular color value of the pixel
//VdotH     = the dot product of the camera view direction and the half vector 
float3 SchlickFresnel(float3 specular, float VdotH)
{
    return specular + (float3(1.0, 1.0, 1.0) - specular) * pow(1.0 - VdotH, 5.0);
}

//Schlick Gaussian Fresnel 
//specular  = the rgb specular color value of the pixel
//VdotH     = the dot product of the camera view direction and the half vector 
float3 SchlickGaussianFresnel(in float3 specular, in float VdotH)
{
    float sphericalGaussian = pow(2.0, (-5.55473 * VdotH - 6.98316) * VdotH);
    return specular + (float3(1.0, 1.0, 1.0) - specular) * sphericalGaussian;
}

float3 SchlickFresnelCustom(float3 specular, float LdotH)
{
    float ior = 0.25;
    float airIor = 1.000277;
    float f0 = (ior - airIor) / (ior + airIor);
    const float max_ior = 2.5;
    f0 = clamp(f0 * f0, 0.0, (max_ior - airIor) / (max_ior + airIor));
    return specular * (f0   + (1 - f0) * pow(2, (-5.55473 * LdotH - 6.98316) * LdotH));
}

//Get Fresnel
//specular  = the rgb specular color value of the pixel
//VdotH     = the dot product of the camera view direction and the half vector 
float3 Fresnel(float3 specular, float VdotH, float LdotH)
{
    return SchlickFresnelCustom(specular, LdotH);
    //return SchlickFresnel(specular, VdotH);
}

// Smith GGX corrected Visibility
// NdotL        = the dot product of the normal and direction to the light
// NdotV        = the dot product of the normal and the camera view direction
// roughness    = the roughness of the pixel
float SmithGGXSchlickVisibility(float NdotL, float NdotV, float roughness)
{
    float rough2 = roughness * roughness;
    float lambdaV = NdotL  * sqrt((-NdotV * rough2 + NdotV) * NdotV + rough2);   
    float lambdaL = NdotV  * sqrt((-NdotL * rough2 + NdotL) * NdotL + rough2);

    return 0.5 / (lambdaV + lambdaL);
}

float NeumannVisibility(float NdotV, float NdotL) 
{
    return NdotL * NdotV / max(1e-7, max(NdotL, NdotV));
}

// Get Visibility
// NdotL        = the dot product of the normal and direction to the light
// NdotV        = the dot product of the normal and the camera view direction
// roughness    = the roughness of the pixel
float Visibility(float NdotL, float NdotV, float roughness)
{
    return NeumannVisibility(NdotV, NdotL);
    //return SmithGGXSchlickVisibility(NdotL, NdotV, roughness);
}

// GGX Distribution
// NdotH        = the dot product of the normal and the half vector
// roughness    = the roughness of the pixel
float GGXDistribution(float NdotH, float roughness)
{
    float rough2 = roughness * roughness;
    float tmp =  (NdotH * rough2 - NdotH) * NdotH + 1;
    return rough2 / (tmp * tmp);
}

// Blinn Distribution
// NdotH        = the dot product of the normal and the half vector
// roughness    = the roughness of the pixel
float BlinnPhongDistribution(in float NdotH, in float roughness)
{
    const float specPower = max((2.0 / (roughness * roughness)) - 2.0, 1e-4f); // Calculate specular power from roughness
    return pow(saturate(NdotH), specPower);
}

// Beckmann Distribution
// NdotH        = the dot product of the normal and the half vector
// roughness    = the roughness of the pixel
float BeckmannDistribution(in float NdotH, in float roughness)
{
    const float rough2 = roughness * roughness;
    const float roughnessA = 1.0 / (4.0 * rough2 * pow(NdotH, 4.0));
    const float roughnessB = NdotH * NdotH - 1.0;
    const float roughnessC = rough2 * NdotH * NdotH;
    return roughnessA * exp(roughnessB / roughnessC);
}

// Get Distribution
// NdotH        = the dot product of the normal and the half vector
// roughness    = the roughness of the pixel
float Distribution(float NdotH, float roughness)
{
    return GGXDistribution(NdotH, roughness);
}

// Lambertian Diffuse
// diffuseColor = the rgb color value of the pixel
// roughness    = the roughness of the pixel
// NdotV        = the normal dot with the camera view direction
// NdotL        = the normal dot with the light direction
// VdotH        = the camera view direction dot with the half vector
float3 LambertianDiffuse(float3 diffuseColor)
{
    return diffuseColor * (1.0 / M_PI) ;
}

// Custom Lambertian Diffuse
// diffuseColor = the rgb color value of the pixel
// roughness    = the roughness of the pixel
// NdotV        = the normal dot with the camera view direction
// NdotL        = the normal dot with the light direction
// VdotH        = the camera view direction dot with the half vector
float3 CustomLambertianDiffuse(float3 diffuseColor, float NdotV, float roughness)
{
    return diffuseColor * (1.0 / M_PI) * pow(NdotV, 0.5 + 0.3 * roughness);
}

// Burley Diffuse
// diffuseColor = the rgb color value of the pixel
// roughness    = the roughness of the pixel
// NdotV        = the normal dot with the camera view direction
// NdotL        = the normal dot with the light direction
// VdotH        = the camera view direction dot with the half vector
float3 BurleyDiffuse(float3 diffuseColor, float roughness, float NdotV, float NdotL, float VdotH)
{
    const float energyBias = lerp(0, 0.5, roughness);
    const float energyFactor = lerp(1.0, 1.0 / 1.51, roughness);
    const float fd90 = energyBias + 2.0 * VdotH * VdotH * roughness;
    const float f0 = 1.0;
    const float lightScatter = f0 + (fd90 - f0) * pow(1.0f - NdotL, 5.0f);
    const float viewScatter = f0 + (fd90 - f0) * pow(1.0f - NdotV, 5.0f);

    return diffuseColor * lightScatter * viewScatter * energyFactor;
}

//Get Diffuse
// diffuseColor = the rgb color value of the pixel
// roughness    = the roughness of the pixel
// NdotV        = the normal dot with the camera view direction
// NdotL        = the normal dot with the light direction
// VdotH        = the camera view direction dot with the half vector
float3 Diffuse(float3 diffuseColor, float roughness, float NdotV, float NdotL, float VdotH)
{
    //return LambertianDiffuse(diffuseColor);
    return CustomLambertianDiffuse(diffuseColor, NdotV, roughness);
    //return BurleyDiffuse(diffuseColor, roughness, NdotV, NdotL, VdotH);
}

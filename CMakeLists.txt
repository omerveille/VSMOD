cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(vsmod)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/vsmod")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Azéline Aillet (Former student at EPITA), Gabriel Husak (Former student at EPITA)")
set(EXTENSION_DESCRIPTION "A 3D Slicer Vessel Segmentation and MODelization plugin.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/vsmod.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/vsmod/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(vsmod)
add_subdirectory(DefaultModule)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})

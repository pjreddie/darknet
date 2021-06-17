
project(YoloCSharpWrapper LANGUAGES CSharp)
include(CSharpUtilities)

add_library(${PROJECT_NAME}
  ${PROJECT_NAME}.cs
)

target_link_libraries(${PROJECT_NAME} PRIVATE dark)

set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DOTNET_REFERENCES
  "System"
  "System.Runtime.InteropServices"
)

install(TARGETS ${PROJECT_NAME}
  RUNTIME DESTINATION "${INSTALL_BIN_DIR}"
  COMPONENT dev
)

function getLocationFromRoute(route) {
  var { _modelType, _id } = route.params;
  if (_modelType) {
    return { _modelType, _id };
  }
  return null;
}

function getPathFromLocation(location) {
  if (!location) {
    return "/";
  }
  return `/${location._modelType || location.type}${
    location._id ? `/${location._id}` : ""
  }`;
}

export { getLocationFromRoute, getPathFromLocation };

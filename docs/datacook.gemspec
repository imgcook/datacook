# coding: utf-8

Gem::Specification.new do |spec|
  spec.name          = "datacook"
  spec.version       = "0.3.3"
  spec.authors       = ["Jan Wang"]
  spec.email         = ["wangxjbuaa@foxmail.com"]

  spec.summary       = %q{A machine learning and data science library for Javascript/Typescript}
  spec.homepage      = "https://github.com/imgcook/datacook"
  spec.license       = "MIT"

  spec.files         = `git ls-files -z`.split("\x0").select { |f| f.match(%r{^(assets|bin|_layouts|_includes|lib|Rakefile|_sass|LICENSE|README)}i) }
  spec.executables   << 'datacook'

  spec.add_development_dependency "bundler", "~> 2.2.23"
  spec.add_runtime_dependency "jekyll", ">= 3.8.5"
  spec.add_runtime_dependency "jekyll-seo-tag", "~> 2.0"
  spec.add_runtime_dependency "rake", ">= 12.3.1", "< 13.1.0"

end

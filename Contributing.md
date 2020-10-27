This style guide extends from the official [Google Javascript style guide](https://google.github.io/styleguide/jsguide.html)

## Commits
Git commits should follow the format:

`[file/method/function changed]: (Fixes #num | Refs #num ) Your descriptive commit message`

__Note:__ #num is a pull or issue number.

For example:

`[Imageprocessor]: Fixes #20 Add crop method`

## File names
File names must be all lowercase and may include underscores (_) or dashes (-). Filenames’ extension must be .js.

## Indentation
We use two spaces for indentation. If you use a code editor like vscode, you can set a default spaces to 2 instead of 4. We do not use Tab.

## Source file structure

Files consist of the following, in order:

 - License or copyright information, if present
 - ES import statements, if an ES module
 - The file’s source code 

 Example:

 ```javascript
/**
* Copyright 2020, Datacook.
* All rights reserved.
*
* This source code is licensed under the MIT license found in the
* LICENSE file in the root directory of this source tree.
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
 *
*/

import table from 'table-data'
import me from 'you'
import { cat, dog, eagle } from '../animals'

/**
 * Returns the sum of two numbers
 * @param {number} num1 
 * @param {number} num2 
 * @returns {number} sum of num1 and num2
 */
 getSum(num1, num2) {
   return num1 + num2

 ```


## Naming Convention

### Class names
Class, interface, record, and typedef names are written in UpperCamelCase e.g `ImageProcessor`.
Type names are typically nouns or noun phrases. For example, Request, ImmutableList, or VisibilityMode.

### Method names
Method names are written in lowerCamelCase e.g `addNum`. Names for private methods must start with a trailing underscore e.g `_startAddition`.

Method names are typically verbs or verb phrases. For example, `sendMessage` or `_stopProcess`. Getter and setter methods for properties are never required, but if they are used they should be named `getFoo` (or optionally `isFoo` or `hasFoo` for booleans), or `setFoo(value)` for setters.

### Constant names
Constant names use `CONSTANT_CASE`: all uppercase letters, with words separated by underscores.


## JSDoc
JSDoc is used on all classes, fields, functions/methods.
The basic formatting of JSDoc blocks is as seen in this example:

```javascript
/**
 * Multiple lines of JSDoc text are written here,
 * wrapped normally.
 * @param {number} arg A number to do something to.
 * @returns {string} name A name of the compute
 */
function doSomething(arg) { … }
```

## Testing

We use [Mocha](https://mochajs.org/) for testing, and encourage contributors to follow a [Test Driven Development](https://en.wikipedia.org/wiki/Test-driven_development) (TDD) approach where you write test to fail at first and then write the corresponding function to pass the test. 
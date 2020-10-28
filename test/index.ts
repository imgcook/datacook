import { hello } from '../src/index';
import { expect } from 'chai';
import 'mocha';

describe('First test', 
  () => { 
    it('should return true', () => { 
      const result = hello('1');
      expect(result).to.equal('1'); 
  }); 
});
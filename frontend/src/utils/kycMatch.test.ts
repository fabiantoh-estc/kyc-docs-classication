import { describe, expect, it } from "vitest";
import { compareKycNames, identityFullName } from "./kycMatch";

describe("customer onboarding KYC name match", () => {
  it("matches reordered passport and utilities names", () => {
    const identity = identityFullName({
      given_names: "ALICE MAY",
      surname: "EXAMPLE",
    });
    const result = compareKycNames(identity, "EXAMPLE ALICE MAY");
    expect(result.status).toBe("match");
    expect(result.matched).toBe(true);
  });

  it("flags a mismatch when surnames differ", () => {
    const result = compareKycNames("BOB SPECIMEN", "EXAMPLE ALICE MAY");
    expect(result.status).toBe("mismatch");
    expect(result.matched).toBe(false);
  });

  it("stays incomplete until both names exist", () => {
    expect(compareKycNames(null, "EXAMPLE ALICE MAY").status).toBe(
      "incomplete",
    );
  });
});

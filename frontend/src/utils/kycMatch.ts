const HONORIFICS = new Set(["MR", "MRS", "MS", "MISS", "MDM", "DR", "SIR"]);

export type KycMatch = {
  matched: boolean;
  status: "match" | "mismatch" | "incomplete";
  reason: string;
  identityName: string | null;
  billName: string | null;
};

export function nameTokens(value: string | null | undefined): string[] {
  if (!value) return [];
  return value
    .normalize("NFKC")
    .toUpperCase()
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .split(/\s+/)
    .filter((token) => token && !HONORIFICS.has(token));
}

export function identityFullName(fields: {
  given_names?: string | null;
  surname?: string | null;
}): string | null {
  const combined = [fields.given_names, fields.surname]
    .filter((part): part is string => Boolean(part && part.trim()))
    .join(" ")
    .trim();
  return combined || null;
}

export function compareKycNames(
  identityName: string | null | undefined,
  billName: string | null | undefined,
): KycMatch {
  const identity = nameTokens(identityName);
  const bill = nameTokens(billName);
  if (!identity.length || !bill.length) {
    return {
      matched: false,
      status: "incomplete",
      reason:
        "Process a passport and a utilities bill to complete customer onboarding KYC.",
      identityName: identityName ?? null,
      billName: billName ?? null,
    };
  }
  const identitySet = new Set(identity);
  const billSet = new Set(bill);
  const overlap = [...identitySet].filter((token) => billSet.has(token));
  const subset =
    overlap.length >= 2 &&
    (identity.every((token) => billSet.has(token)) ||
      bill.every((token) => identitySet.has(token)));
  if (
    identity.length === bill.length &&
    identity.every((token) => billSet.has(token))
  ) {
    return {
      matched: true,
      status: "match",
      reason: "Account holder name matches the identity document.",
      identityName: identityName ?? null,
      billName: billName ?? null,
    };
  }
  if (subset || overlap.length >= 2) {
    return {
      matched: true,
      status: "match",
      reason: "Account holder name matches the identity document.",
      identityName: identityName ?? null,
      billName: billName ?? null,
    };
  }
  return {
    matched: false,
    status: "mismatch",
    reason: "Account holder name does not match the identity document.",
    identityName: identityName ?? null,
    billName: billName ?? null,
  };
}
